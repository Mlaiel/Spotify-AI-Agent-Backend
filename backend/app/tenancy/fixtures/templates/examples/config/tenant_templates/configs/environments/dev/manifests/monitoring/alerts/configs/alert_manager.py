"""
Gestionnaire Central des Alertes - Spotify AI Agent
===================================================

Gestionnaire intelligent des alertes multi-tenant avec capacités avancées
de corrélation, machine learning et escalade automatique.

Fonctionnalités:
- Gestion centralisée des alertes multi-tenant
- Corrélation intelligente avec ML
- Escalade automatique basée sur les SLA
- Suppression du bruit et déduplication
- Intégration temps réel avec les systèmes externes
- Analytics et reporting avancés
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import yaml

from .correlation_engine import CorrelationEngine
from .escalation_manager import EscalationManager
from .notification_dispatcher import NotificationDispatcher
from .metrics_collector import MetricsCollector


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes avec scoring ML"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """États des alertes dans le cycle de vie"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"
    ESCALATED = "escalated"


@dataclass
class Alert:
    """Structure complète d'une alerte avec métadonnées étendues"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    status: AlertStatus = AlertStatus.FIRING
    source: str = ""
    service: str = ""
    environment: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    escalation_level: int = 0
    sla_deadline: Optional[datetime] = None
    ml_confidence: float = 0.0
    predicted_impact: str = ""
    similar_alerts: List[str] = field(default_factory=list)
    resolution_suggestions: List[str] = field(default_factory=list)


class AlertManager:
    """Gestionnaire avancé des alertes avec IA et ML intégré"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Composants intégrés
        self.correlation_engine = CorrelationEngine(config.get('correlation', {}))
        self.escalation_manager = EscalationManager(config.get('escalation', {}))
        self.notification_dispatcher = NotificationDispatcher(config.get('notifications', {}))
        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        
        # Stockage et cache
        self.redis_client = None
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.silence_rules: Dict[str, Dict] = {}
        
        # ML et Analytics
        self.alert_patterns: Dict[str, Any] = {}
        self.ml_models = {}
        self.scaler = StandardScaler()
        
        # Métriques Prometheus
        self.metrics = {
            'alerts_total': Counter('alerts_total', 'Total alerts', ['tenant_id', 'severity', 'service']),
            'alerts_duration': Histogram('alert_duration_seconds', 'Alert duration', ['tenant_id', 'severity']),
            'escalations_total': Counter('escalations_total', 'Total escalations', ['tenant_id', 'level']),
            'active_alerts': Gauge('active_alerts', 'Currently active alerts', ['tenant_id', 'severity'])
        }
        
        # Configuration des seuils adaptatifs
        self.adaptive_thresholds = defaultdict(lambda: defaultdict(float))
        self.threshold_learning_window = timedelta(hours=24)
        
        # Gestionnaire d'événements
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
    async def initialize(self):
        """Initialisation asynchrone du gestionnaire"""
        try:
            # Connexion Redis pour la persistance
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Chargement des données historiques
            await self._load_historical_data()
            
            # Initialisation des modèles ML
            await self._initialize_ml_models()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._background_correlation())
            asyncio.create_task(self._adaptive_threshold_learning())
            asyncio.create_task(self._periodic_cleanup())
            
            self.logger.info("AlertManager initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Traitement intelligent d'une nouvelle alerte"""
        try:
            # Création de l'alerte
            alert = Alert(
                tenant_id=alert_data.get('tenant_id', ''),
                title=alert_data.get('title', ''),
                description=alert_data.get('description', ''),
                severity=AlertSeverity(alert_data.get('severity', 'info')),
                source=alert_data.get('source', ''),
                service=alert_data.get('service', ''),
                environment=alert_data.get('environment', ''),
                labels=alert_data.get('labels', {}),
                annotations=alert_data.get('annotations', {}),
                metrics=alert_data.get('metrics', {})
            )
            
            # Enrichissement avec ML
            await self._enrich_alert_with_ml(alert)
            
            # Vérification des règles de silence
            if await self._is_silenced(alert):
                alert.status = AlertStatus.SILENCED
                return alert
            
            # Déduplication intelligente
            existing_alert = await self._find_duplicate_alert(alert)
            if existing_alert:
                await self._merge_alerts(existing_alert, alert)
                return existing_alert
            
            # Corrélation avec les alertes existantes
            correlation_id = await self.correlation_engine.correlate_alert(alert)
            if correlation_id:
                alert.correlation_id = correlation_id
            
            # Calcul du SLA et escalade
            alert.sla_deadline = await self._calculate_sla_deadline(alert)
            
            # Stockage et indexation
            self.active_alerts[alert.id] = alert
            await self._persist_alert(alert)
            
            # Notifications
            await self.notification_dispatcher.dispatch_alert(alert)
            
            # Métriques
            self._update_metrics(alert)
            
            # Événements
            await self._trigger_event('alert_created', alert)
            
            self.logger.info(f"Alerte traitée: {alert.id} - {alert.title}")
            return alert
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de l'alerte: {e}")
            raise
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = None, resolution_note: str = None):
        """Résolution d'une alerte avec enrichissement automatique"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alerte {alert_id} non trouvée")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        if resolution_note:
            alert.annotations['resolution_note'] = resolution_note
        if resolved_by:
            alert.annotations['resolved_by'] = resolved_by
        
        # Calcul de la durée pour les métriques
        duration = (alert.resolved_at - alert.timestamp).total_seconds()
        self.metrics['alerts_duration'].labels(
            tenant_id=alert.tenant_id,
            severity=alert.severity.value
        ).observe(duration)
        
        # Apprentissage ML sur la résolution
        await self._learn_from_resolution(alert)
        
        # Archivage
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        await self._persist_alert(alert)
        await self._trigger_event('alert_resolved', alert)
        
        self.logger.info(f"Alerte résolue: {alert_id}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acquittement d'une alerte"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alerte {alert_id} non trouvée")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by
        
        await self._persist_alert(alert)
        await self._trigger_event('alert_acknowledged', alert)
        
        self.logger.info(f"Alerte acquittée: {alert_id} par {acknowledged_by}")
    
    async def create_silence_rule(self, rule: Dict[str, Any]) -> str:
        """Création d'une règle de silence intelligente"""
        rule_id = str(uuid.uuid4())
        rule['id'] = rule_id
        rule['created_at'] = datetime.utcnow().isoformat()
        
        self.silence_rules[rule_id] = rule
        await self.redis_client.hset('silence_rules', rule_id, json.dumps(rule))
        
        self.logger.info(f"Règle de silence créée: {rule_id}")
        return rule_id
    
    async def get_tenant_dashboard_data(self, tenant_id: str) -> Dict[str, Any]:
        """Génération des données de dashboard pour un tenant"""
        tenant_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.tenant_id == tenant_id
        ]
        
        # Statistiques de base
        stats = {
            'total_active': len(tenant_alerts),
            'by_severity': defaultdict(int),
            'by_service': defaultdict(int),
            'trends': await self._calculate_alert_trends(tenant_id),
            'top_issues': await self._get_top_issues(tenant_id),
            'sla_status': await self._get_sla_status(tenant_id),
            'ml_insights': await self._get_ml_insights(tenant_id)
        }
        
        for alert in tenant_alerts:
            stats['by_severity'][alert.severity.value] += 1
            stats['by_service'][alert.service] += 1
        
        return stats
    
    async def _enrich_alert_with_ml(self, alert: Alert):
        """Enrichissement d'alerte avec ML et prédictions"""
        try:
            # Prédiction de l'impact
            if 'impact_model' in self.ml_models:
                features = self._extract_alert_features(alert)
                predicted_impact = self.ml_models['impact_model'].predict([features])[0]
                alert.predicted_impact = predicted_impact
            
            # Calcul de la confiance ML
            alert.ml_confidence = await self._calculate_ml_confidence(alert)
            
            # Suggestions de résolution basées sur l'historique
            alert.resolution_suggestions = await self._generate_resolution_suggestions(alert)
            
            # Recherche d'alertes similaires
            alert.similar_alerts = await self._find_similar_alerts(alert)
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'enrichissement ML: {e}")
    
    async def _find_duplicate_alert(self, alert: Alert) -> Optional[Alert]:
        """Détection intelligente des doublons d'alertes"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.tenant_id == alert.tenant_id and
                existing_alert.service == alert.service and
                existing_alert.title == alert.title and
                existing_alert.status in [AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED]):
                
                # Vérification de la similitude temporelle (5 minutes)
                time_diff = abs((alert.timestamp - existing_alert.timestamp).total_seconds())
                if time_diff < 300:
                    return existing_alert
        
        return None
    
    async def _calculate_sla_deadline(self, alert: Alert) -> datetime:
        """Calcul intelligent des deadlines SLA"""
        base_sla = {
            AlertSeverity.CRITICAL: timedelta(minutes=15),
            AlertSeverity.HIGH: timedelta(hours=1),
            AlertSeverity.MEDIUM: timedelta(hours=4),
            AlertSeverity.LOW: timedelta(hours=24),
            AlertSeverity.INFO: timedelta(days=3)
        }
        
        sla_duration = base_sla.get(alert.severity, timedelta(hours=4))
        
        # Ajustement basé sur le tenant et le service
        tenant_config = self.config.get('tenants', {}).get(alert.tenant_id, {})
        service_config = tenant_config.get('services', {}).get(alert.service, {})
        
        if 'sla_multiplier' in service_config:
            sla_duration *= service_config['sla_multiplier']
        
        return alert.timestamp + sla_duration
    
    async def _background_correlation(self):
        """Tâche de fond pour la corrélation continue"""
        while True:
            try:
                await asyncio.sleep(60)  # Corrélation toutes les minutes
                await self.correlation_engine.analyze_active_alerts(
                    list(self.active_alerts.values())
                )
            except Exception as e:
                self.logger.error(f"Erreur dans la corrélation de fond: {e}")
    
    async def _adaptive_threshold_learning(self):
        """Apprentissage adaptatif des seuils"""
        while True:
            try:
                await asyncio.sleep(3600)  # Apprentissage toutes les heures
                await self._update_adaptive_thresholds()
            except Exception as e:
                self.logger.error(f"Erreur dans l'apprentissage adaptatif: {e}")
    
    async def _periodic_cleanup(self):
        """Nettoyage périodique des données"""
        while True:
            try:
                await asyncio.sleep(86400)  # Nettoyage quotidien
                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Erreur lors du nettoyage: {e}")
    
    def _update_metrics(self, alert: Alert):
        """Mise à jour des métriques Prometheus"""
        self.metrics['alerts_total'].labels(
            tenant_id=alert.tenant_id,
            severity=alert.severity.value,
            service=alert.service
        ).inc()
        
        self.metrics['active_alerts'].labels(
            tenant_id=alert.tenant_id,
            severity=alert.severity.value
        ).inc()
    
    async def _trigger_event(self, event_type: str, alert: Alert):
        """Déclenchement d'événements pour les handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"Erreur dans le handler {handler}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Enregistrement d'un handler d'événement"""
        self.event_handlers[event_type].append(handler)
    
    @classmethod
    async def from_config(cls, config_path: str) -> 'AlertManager':
        """Factory method pour créer un AlertManager depuis un fichier de config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        manager = cls(config)
        await manager.initialize()
        return manager
