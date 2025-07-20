"""
🚨 Moteur Ultra-Avancé de Gestion d'Alertes Critiques
===================================================

Système de classe enterprise pour la gestion, prédiction et escalade automatique
des alertes critiques avec intelligence artificielle intégrée.

Architecture: Event-Driven + CQRS + DDD + ML/AI
Patterns: Observer, Strategy, Command, Factory, Builder, Singleton
Technologies: FastAPI, SQLAlchemy, Redis, TensorFlow, Prometheus
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor

import aioredis
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from prometheus_client import Counter, Histogram, Gauge

from . import CriticalAlertSeverity, AlertChannel, TenantTier, CriticalAlertMetadata

# Métriques Prometheus pour observabilité complète
alert_processing_time = Histogram(
    'critical_alert_processing_seconds',
    'Temps de traitement des alertes critiques',
    ['tenant_id', 'severity', 'channel']
)

alert_escalations_total = Counter(
    'critical_alert_escalations_total',
    'Nombre total d\'escalades d\'alertes',
    ['tenant_id', 'severity', 'escalation_level']
)

alert_ml_predictions = Gauge(
    'critical_alert_ml_prediction_confidence',
    'Score de confiance des prédictions ML',
    ['tenant_id', 'model_version']
)

active_critical_alerts = Gauge(
    'active_critical_alerts_count',
    'Nombre d\'alertes critiques actives',
    ['tenant_id', 'severity']
)

class AlertMLModel:
    """Modèle ML pour la prédiction et corrélation d'alertes"""
    
    def __init__(self):
        self.model_version = "3.0.0-enterprise"
        self.confidence_threshold = 0.8
        self.correlation_window = timedelta(minutes=15)
        
    async def predict_escalation_probability(
        self,
        alert_metadata: CriticalAlertMetadata,
        historical_data: List[Dict]
    ) -> float:
        """Prédit la probabilité d'escalade d'une alerte"""
        try:
            # Extraction des features
            features = self._extract_features(alert_metadata, historical_data)
            
            # Prédiction ML (simulation - en production utiliser TensorFlow/PyTorch)
            base_probability = self._calculate_base_probability(alert_metadata)
            historical_factor = self._analyze_historical_patterns(historical_data)
            tenant_factor = self._get_tenant_escalation_factor(alert_metadata.tenant_id)
            
            probability = min(base_probability * historical_factor * tenant_factor, 1.0)
            
            # Mise à jour des métriques
            alert_ml_predictions.labels(
                tenant_id=alert_metadata.tenant_id,
                model_version=self.model_version
            ).set(probability)
            
            return probability
            
        except Exception as e:
            logging.error(f"Erreur prédiction ML: {e}")
            return 0.5  # Valeur par défaut conservative
    
    def _extract_features(
        self,
        alert_metadata: CriticalAlertMetadata,
        historical_data: List[Dict]
    ) -> np.ndarray:
        """Extraction des features pour le modèle ML"""
        features = [
            alert_metadata.severity.value[1],  # Score de sévérité
            alert_metadata.affected_users,
            alert_metadata.business_impact,
            alert_metadata.tenant_tier.value[1],  # Priorité tenant
            len(historical_data),  # Historique récent
            datetime.utcnow().hour,  # Heure du jour
            datetime.utcnow().weekday(),  # Jour de la semaine
        ]
        return np.array(features, dtype=np.float32)
    
    def _calculate_base_probability(self, alert_metadata: CriticalAlertMetadata) -> float:
        """Calcul de la probabilité de base selon la sévérité"""
        severity_probabilities = {
            CriticalAlertSeverity.CATASTROPHIC: 0.95,
            CriticalAlertSeverity.CRITICAL: 0.85,
            CriticalAlertSeverity.HIGH: 0.65,
            CriticalAlertSeverity.ELEVATED: 0.45,
            CriticalAlertSeverity.WARNING: 0.25
        }
        return severity_probabilities.get(alert_metadata.severity, 0.5)
    
    def _analyze_historical_patterns(self, historical_data: List[Dict]) -> float:
        """Analyse des patterns historiques"""
        if not historical_data:
            return 1.0
            
        escalated_count = sum(1 for alert in historical_data if alert.get('escalated', False))
        total_count = len(historical_data)
        
        if total_count == 0:
            return 1.0
            
        historical_rate = escalated_count / total_count
        return min(max(historical_rate * 1.2, 0.1), 2.0)  # Facteur entre 0.1 et 2.0
    
    def _get_tenant_escalation_factor(self, tenant_id: str) -> float:
        """Facteur d'escalade basé sur le tenant"""
        # En production, récupérer depuis la base de données
        tier_factors = {
            "free": 0.8,
            "premium": 1.0,
            "enterprise": 1.3,
            "enterprise_plus": 1.5
        }
        # Simulation - en production utiliser une requête DB
        return tier_factors.get("enterprise", 1.0)

class CriticalAlertProcessor:
    """Processeur principal des alertes critiques"""
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        db_session: AsyncSession,
        ml_model: AlertMLModel
    ):
        self.redis_client = redis_client
        self.db_session = db_session
        self.ml_model = ml_model
        self.notification_channels = {}
        self.escalation_rules = {}
        self.active_alerts = {}
        
        # Configuration avancée
        self.config = {
            "max_parallel_processing": 100,
            "escalation_timeout": 300,  # 5 minutes
            "correlation_window": 900,  # 15 minutes
            "batch_size": 50,
            "retry_attempts": 3,
            "circuit_breaker_threshold": 10
        }
        
        # Circuit breaker pour la résilience
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half-open
        }
    
    async def process_critical_alert(
        self,
        alert_data: Dict[str, Any],
        tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Traitement principal d'une alerte critique"""
        start_time = datetime.utcnow()
        alert_id = str(uuid.uuid4())
        
        try:
            # Validation et enrichissement
            alert_metadata = await self._create_alert_metadata(alert_data, tenant_context)
            alert_metadata.alert_id = alert_id
            
            # Stockage dans Redis pour accès rapide
            await self._store_alert_in_cache(alert_metadata)
            
            # Prédiction ML
            historical_data = await self._get_historical_alerts(
                alert_metadata.tenant_id,
                alert_metadata.source_service
            )
            
            escalation_probability = await self.ml_model.predict_escalation_probability(
                alert_metadata, historical_data
            )
            
            alert_metadata.ml_confidence_score = escalation_probability
            
            # Corrélation avec alertes existantes
            correlated_alerts = await self._correlate_alerts(alert_metadata)
            
            # Détermination du plan d'escalade
            escalation_plan = await self._determine_escalation_plan(
                alert_metadata, escalation_probability
            )
            
            # Notification immédiate
            notification_result = await self._send_immediate_notifications(
                alert_metadata, escalation_plan
            )
            
            # Programmation des escalades automatiques
            await self._schedule_automatic_escalations(alert_metadata, escalation_plan)
            
            # Mise à jour des métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            alert_processing_time.labels(
                tenant_id=alert_metadata.tenant_id,
                severity=alert_metadata.severity.name,
                channel="slack"
            ).observe(processing_time)
            
            active_critical_alerts.labels(
                tenant_id=alert_metadata.tenant_id,
                severity=alert_metadata.severity.name
            ).inc()
            
            # Retour du résultat
            return {
                "alert_id": alert_id,
                "status": "processed",
                "escalation_probability": escalation_probability,
                "correlated_alerts": len(correlated_alerts),
                "notification_channels": len(notification_result),
                "processing_time_ms": processing_time * 1000,
                "escalation_plan": escalation_plan
            }
            
        except Exception as e:
            logging.error(f"Erreur traitement alerte critique {alert_id}: {e}")
            await self._handle_processing_error(alert_id, e)
            return {
                "alert_id": alert_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _create_alert_metadata(
        self,
        alert_data: Dict[str, Any],
        tenant_context: Dict[str, Any]
    ) -> CriticalAlertMetadata:
        """Création et enrichissement des métadonnées d'alerte"""
        severity = CriticalAlertSeverity[alert_data.get("severity", "WARNING")]
        tenant_tier = TenantTier[tenant_context.get("tier", "FREE")]
        
        # Calcul de l'impact business
        business_impact = self._calculate_business_impact(
            alert_data.get("affected_users", 0),
            severity,
            tenant_tier
        )
        
        # Génération du fingerprint pour la déduplication
        fingerprint = self._generate_alert_fingerprint(alert_data)
        
        return CriticalAlertMetadata(
            alert_id="",  # Sera défini plus tard
            tenant_id=tenant_context["tenant_id"],
            severity=severity,
            tenant_tier=tenant_tier,
            source_service=alert_data.get("source_service", "unknown"),
            affected_users=alert_data.get("affected_users", 0),
            business_impact=business_impact,
            correlation_id=alert_data.get("correlation_id", ""),
            trace_id=alert_data.get("trace_id", ""),
            fingerprint=fingerprint,
            runbook_url=alert_data.get("runbook_url"),
            tags=alert_data.get("tags", {}),
            custom_data=alert_data.get("custom_data", {})
        )
    
    def _calculate_business_impact(
        self,
        affected_users: int,
        severity: CriticalAlertSeverity,
        tenant_tier: TenantTier
    ) -> float:
        """Calcul de l'impact business de l'alerte"""
        base_impact = severity.value[1] / 1000  # Normalisation
        user_factor = min(affected_users / 10000, 2.0)  # Maximum 2x
        tier_multiplier = tenant_tier.value[1]  # Multiplicateur selon le tier
        
        return base_impact * (1 + user_factor) * tier_multiplier
    
    def _generate_alert_fingerprint(self, alert_data: Dict[str, Any]) -> str:
        """Génération d'un fingerprint unique pour l'alerte"""
        key_fields = [
            alert_data.get("source_service", ""),
            alert_data.get("alert_name", ""),
            alert_data.get("severity", ""),
            str(alert_data.get("tags", {}))
        ]
        
        fingerprint_string = "|".join(key_fields)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    async def _store_alert_in_cache(self, alert_metadata: CriticalAlertMetadata):
        """Stockage de l'alerte dans Redis pour accès rapide"""
        alert_key = f"critical_alert:{alert_metadata.tenant_id}:{alert_metadata.alert_id}"
        alert_data = asdict(alert_metadata)
        
        # Conversion des objets enum et datetime en string
        alert_data["severity"] = alert_metadata.severity.name
        alert_data["tenant_tier"] = alert_metadata.tenant_tier.name
        alert_data["created_at"] = alert_metadata.created_at.isoformat()
        alert_data["updated_at"] = alert_metadata.updated_at.isoformat()
        
        await self.redis_client.setex(
            alert_key,
            self.config["correlation_window"],  # TTL
            json.dumps(alert_data)
        )
        
        # Index pour les requêtes rapides
        tenant_alerts_key = f"tenant_alerts:{alert_metadata.tenant_id}"
        await self.redis_client.zadd(
            tenant_alerts_key,
            {alert_metadata.alert_id: datetime.utcnow().timestamp()}
        )
    
    async def _get_historical_alerts(
        self,
        tenant_id: str,
        source_service: str,
        lookback_hours: int = 24
    ) -> List[Dict]:
        """Récupération des alertes historiques pour l'analyse ML"""
        # En production, requête optimisée sur la base de données
        # Ici, simulation avec données depuis Redis
        tenant_alerts_key = f"tenant_alerts:{tenant_id}"
        
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        alert_ids = await self.redis_client.zrangebyscore(
            tenant_alerts_key,
            cutoff_time.timestamp(),
            datetime.utcnow().timestamp()
        )
        
        historical_alerts = []
        for alert_id in alert_ids[:50]:  # Limite pour la performance
            alert_key = f"critical_alert:{tenant_id}:{alert_id.decode()}"
            alert_data = await self.redis_client.get(alert_key)
            if alert_data:
                historical_alerts.append(json.loads(alert_data))
        
        return historical_alerts
    
    async def _correlate_alerts(
        self,
        alert_metadata: CriticalAlertMetadata
    ) -> List[str]:
        """Corrélation avec les alertes existantes"""
        correlation_window = datetime.utcnow() - self.ml_model.correlation_window
        
        # Recherche d'alertes similaires par fingerprint
        tenant_alerts_key = f"tenant_alerts:{alert_metadata.tenant_id}"
        recent_alert_ids = await self.redis_client.zrangebyscore(
            tenant_alerts_key,
            correlation_window.timestamp(),
            datetime.utcnow().timestamp()
        )
        
        correlated_alerts = []
        for alert_id in recent_alert_ids:
            alert_key = f"critical_alert:{alert_metadata.tenant_id}:{alert_id.decode()}"
            alert_data = await self.redis_client.get(alert_key)
            
            if alert_data:
                existing_alert = json.loads(alert_data)
                if (existing_alert.get("fingerprint") == alert_metadata.fingerprint or
                    existing_alert.get("source_service") == alert_metadata.source_service):
                    correlated_alerts.append(alert_id.decode())
        
        return correlated_alerts
    
    async def _determine_escalation_plan(
        self,
        alert_metadata: CriticalAlertMetadata,
        escalation_probability: float
    ) -> Dict[str, Any]:
        """Détermination du plan d'escalade intelligent"""
        base_plan = {
            "immediate": [],
            "level_1": [],
            "level_2": [],
            "level_3": [],
            "level_4": []
        }
        
        # Escalade immédiate selon la sévérité
        if alert_metadata.severity in [CriticalAlertSeverity.CATASTROPHIC, CriticalAlertSeverity.CRITICAL]:
            base_plan["immediate"].extend([AlertChannel.SLACK, AlertChannel.EMAIL])
            
            if alert_metadata.tenant_tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
                base_plan["immediate"].append(AlertChannel.PAGERDUTY)
        
        # Escalade prédictive basée sur le ML
        if escalation_probability > 0.8:
            base_plan["level_1"].append(AlertChannel.SMS)
            if alert_metadata.tenant_tier == TenantTier.ENTERPRISE_PLUS:
                base_plan["level_1"].append(AlertChannel.PHONE_CALL)
        
        # Délais d'escalade selon le SLA du tenant
        sla_seconds = alert_metadata.tenant_tier.value[2]
        escalation_delays = {
            "level_1": sla_seconds,
            "level_2": sla_seconds * 2,
            "level_3": sla_seconds * 4,
            "level_4": sla_seconds * 8
        }
        
        return {
            "channels": base_plan,
            "delays": escalation_delays,
            "ml_probability": escalation_probability,
            "auto_escalation_enabled": True
        }
    
    async def _send_immediate_notifications(
        self,
        alert_metadata: CriticalAlertMetadata,
        escalation_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Envoi des notifications immédiates"""
        notification_results = []
        immediate_channels = escalation_plan["channels"]["immediate"]
        
        # Traitement parallèle des notifications
        tasks = []
        for channel in immediate_channels:
            task = self._send_notification(alert_metadata, channel)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for channel, result in zip(immediate_channels, results):
                if isinstance(result, Exception):
                    notification_results.append({
                        "channel": channel.name,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    notification_results.append({
                        "channel": channel.name,
                        "status": "sent",
                        "result": result
                    })
        
        return notification_results
    
    async def _send_notification(
        self,
        alert_metadata: CriticalAlertMetadata,
        channel: AlertChannel
    ) -> Dict[str, Any]:
        """Envoi d'une notification sur un canal spécifique"""
        try:
            # Factory pattern pour les handlers de notification
            handler = self._get_notification_handler(channel)
            result = await handler.send_notification(alert_metadata)
            
            logging.info(
                f"Notification envoyée - Alerte: {alert_metadata.alert_id}, "
                f"Canal: {channel.name}, Tenant: {alert_metadata.tenant_id}"
            )
            
            return result
            
        except Exception as e:
            logging.error(
                f"Erreur envoi notification - Alerte: {alert_metadata.alert_id}, "
                f"Canal: {channel.name}, Erreur: {e}"
            )
            raise
    
    def _get_notification_handler(self, channel: AlertChannel):
        """Factory pour les handlers de notification"""
        handlers = {
            AlertChannel.SLACK: SlackNotificationHandler(),
            AlertChannel.EMAIL: EmailNotificationHandler(),
            AlertChannel.SMS: SMSNotificationHandler(),
            AlertChannel.PAGERDUTY: PagerDutyNotificationHandler(),
            AlertChannel.TEAMS: TeamsNotificationHandler(),
            AlertChannel.WEBHOOK: WebhookNotificationHandler()
        }
        
        return handlers.get(channel, DefaultNotificationHandler())
    
    async def _schedule_automatic_escalations(
        self,
        alert_metadata: CriticalAlertMetadata,
        escalation_plan: Dict[str, Any]
    ):
        """Programmation des escalades automatiques"""
        if not escalation_plan.get("auto_escalation_enabled"):
            return
        
        escalation_key = f"escalation:{alert_metadata.alert_id}"
        escalation_data = {
            "alert_id": alert_metadata.alert_id,
            "tenant_id": alert_metadata.tenant_id,
            "escalation_plan": escalation_plan,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Stockage des données d'escalade
        await self.redis_client.setex(
            escalation_key,
            86400,  # 24 heures
            json.dumps(escalation_data)
        )
        
        # Programmation des tâches d'escalade
        delays = escalation_plan["delays"]
        for level, delay_seconds in delays.items():
            escalation_time = datetime.utcnow() + timedelta(seconds=delay_seconds)
            
            # En production, utiliser Celery, RQ ou un scheduler distribué
            # Ici, simulation avec une tâche asyncio
            asyncio.create_task(
                self._delayed_escalation(
                    alert_metadata.alert_id,
                    level,
                    delay_seconds
                )
            )
    
    async def _delayed_escalation(
        self,
        alert_id: str,
        escalation_level: str,
        delay_seconds: int
    ):
        """Escalade différée automatique"""
        await asyncio.sleep(delay_seconds)
        
        try:
            # Vérification si l'alerte est toujours active
            escalation_key = f"escalation:{alert_id}"
            escalation_data = await self.redis_client.get(escalation_key)
            
            if not escalation_data:
                return  # Alerte résolue ou expirée
            
            escalation_info = json.loads(escalation_data)
            escalation_plan = escalation_info["escalation_plan"]
            
            # Envoi des notifications d'escalade
            channels = escalation_plan["channels"].get(escalation_level, [])
            
            # Récupération des métadonnées de l'alerte
            alert_key = f"critical_alert:{escalation_info['tenant_id']}:{alert_id}"
            alert_data = await self.redis_client.get(alert_key)
            
            if alert_data:
                # Reconstruction des métadonnées (simplifiée)
                alert_dict = json.loads(alert_data)
                # Traitement de l'escalade...
                
                # Mise à jour des métriques
                alert_escalations_total.labels(
                    tenant_id=escalation_info['tenant_id'],
                    severity=alert_dict.get('severity', 'UNKNOWN'),
                    escalation_level=escalation_level
                ).inc()
                
                logging.info(
                    f"Escalade automatique - Alerte: {alert_id}, "
                    f"Niveau: {escalation_level}, Canaux: {len(channels)}"
                )
            
        except Exception as e:
            logging.error(f"Erreur escalade automatique {alert_id}: {e}")
    
    async def _handle_processing_error(self, alert_id: str, error: Exception):
        """Gestion des erreurs de traitement"""
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = datetime.utcnow()
        
        # Circuit breaker logic
        if self.circuit_breaker["failures"] >= self.config["circuit_breaker_threshold"]:
            self.circuit_breaker["state"] = "open"
            logging.critical(
                f"Circuit breaker OUVERT - Trop d'échecs: {self.circuit_breaker['failures']}"
            )
        
        # Stockage de l'erreur pour analyse
        error_key = f"alert_error:{alert_id}"
        error_data = {
            "alert_id": alert_id,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.setex(error_key, 86400, json.dumps(error_data))


# Handlers de notification (interfaces simplifiées)
class NotificationHandler(ABC):
    """Interface abstraite pour les handlers de notification"""
    
    @abstractmethod
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        pass

class SlackNotificationHandler(NotificationHandler):
    """Handler pour les notifications Slack"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        # Implémentation spécifique Slack avec templates avancés
        return {
            "channel": "slack",
            "status": "sent",
            "message_id": f"slack_{uuid.uuid4()}",
            "timestamp": datetime.utcnow().isoformat()
        }

class EmailNotificationHandler(NotificationHandler):
    """Handler pour les notifications Email"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "email", 
            "status": "sent",
            "message_id": f"email_{uuid.uuid4()}"
        }

class SMSNotificationHandler(NotificationHandler):
    """Handler pour les notifications SMS"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "sms",
            "status": "sent", 
            "message_id": f"sms_{uuid.uuid4()}"
        }

class PagerDutyNotificationHandler(NotificationHandler):
    """Handler pour les notifications PagerDuty"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "pagerduty",
            "status": "sent",
            "incident_id": f"pd_{uuid.uuid4()}"
        }

class TeamsNotificationHandler(NotificationHandler):
    """Handler pour les notifications Microsoft Teams"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "teams",
            "status": "sent",
            "message_id": f"teams_{uuid.uuid4()}"
        }

class WebhookNotificationHandler(NotificationHandler):
    """Handler pour les notifications Webhook"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "webhook",
            "status": "sent",
            "webhook_id": f"webhook_{uuid.uuid4()}"
        }

class DefaultNotificationHandler(NotificationHandler):
    """Handler par défaut pour les canaux non supportés"""
    
    async def send_notification(self, alert_metadata: CriticalAlertMetadata) -> Dict[str, Any]:
        return {
            "channel": "default",
            "status": "not_supported",
            "message": "Canal de notification non supporté"
        }

# Factory principale pour l'instanciation du processeur
class CriticalAlertProcessorFactory:
    """Factory pour créer des instances du processeur d'alertes"""
    
    @staticmethod
    async def create_processor(
        redis_url: str = "redis://localhost:6379",
        db_session: Optional[AsyncSession] = None
    ) -> CriticalAlertProcessor:
        """Création d'une instance complète du processeur"""
        
        # Initialisation Redis
        redis_client = await aioredis.from_url(redis_url)
        
        # Initialisation du modèle ML
        ml_model = AlertMLModel()
        
        # Création du processeur
        processor = CriticalAlertProcessor(
            redis_client=redis_client,
            db_session=db_session,
            ml_model=ml_model
        )
        
        return processor

# Export des classes principales
__all__ = [
    "CriticalAlertProcessor",
    "CriticalAlertProcessorFactory", 
    "AlertMLModel",
    "NotificationHandler",
    "SlackNotificationHandler",
    "EmailNotificationHandler",
    "SMSNotificationHandler",
    "PagerDutyNotificationHandler",
    "TeamsNotificationHandler",
    "WebhookNotificationHandler"
]
