#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Monitoring & Alert Management System
============================================

Ultra-advanced real-time monitoring system with intelligent alerting,
predictive analytics, automated remediation, and enterprise-grade
notification orchestration.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import asyncio
import json
import logging
import os
import sys
import time
import smtplib
import ssl
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import statistics
import math
import threading
import queue
import re

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import (
    EnterpriseMetricsSystem, MetricDataPoint, MetricType, 
    MetricCategory, MetricSeverity, get_metrics_system
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import slack_sdk
    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False

try:
    import discord
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AlertPriority(Enum):
    """Priorité des alertes."""
    EMERGENCY = "emergency"      # P0 - Intervention immédiate
    CRITICAL = "critical"        # P1 - Résolution dans l'heure
    HIGH = "high"               # P2 - Résolution dans les 4h
    MEDIUM = "medium"           # P3 - Résolution dans la journée
    LOW = "low"                 # P4 - Résolution dans la semaine
    INFO = "info"               # Information seulement


class AlertStatus(Enum):
    """Statut des alertes."""
    TRIGGERED = "triggered"      # Déclenchée
    ACKNOWLEDGED = "acknowledged" # Acquittée
    INVESTIGATING = "investigating" # En cours d'investigation
    RESOLVING = "resolving"      # En cours de résolution
    RESOLVED = "resolved"        # Résolue
    CLOSED = "closed"           # Fermée
    SUPPRESSED = "suppressed"    # Supprimée


class NotificationChannel(Enum):
    """Canaux de notification."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    PHONE = "phone"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Règle d'alerte avancée."""
    
    # Identification
    rule_id: str
    name: str
    description: str
    
    # Conditions
    metric_pattern: str           # Pattern de métrique (regex)
    threshold_value: float        # Valeur seuil
    comparison: str              # >, <, >=, <=, ==, !=
    duration_seconds: int = 300   # Durée avant déclenchement
    
    # Classification
    priority: AlertPriority = AlertPriority.MEDIUM
    category: MetricCategory = MetricCategory.SYSTEM
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Comportement
    enabled: bool = True
    auto_resolve: bool = True
    auto_resolve_duration: int = 600
    suppress_duration: int = 3600  # Durée de suppression après résolution
    
    # Notification
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_template: str = ""
    escalation_rules: List[str] = field(default_factory=list)  # IDs des règles d'escalade
    
    # Machine Learning
    use_anomaly_detection: bool = False
    ml_sensitivity: float = 0.8   # 0.0 à 1.0
    baseline_window_hours: int = 24
    
    # Conditions avancées
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class Alert:
    """Instance d'alerte."""
    
    # Identification
    alert_id: str
    rule_id: str
    correlation_id: str = ""
    
    # Détails
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    
    # Classification
    priority: AlertPriority
    status: AlertStatus = AlertStatus.TRIGGERED
    category: MetricCategory = MetricCategory.SYSTEM
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Contexte
    affected_hosts: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Notifications
    notifications_sent: List[str] = field(default_factory=list)
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Résolution
    resolution_notes: str = ""
    root_cause: str = ""
    remediation_actions: List[str] = field(default_factory=list)


@dataclass
class MonitoringTarget:
    """Cible de monitoring."""
    
    target_id: str
    name: str
    target_type: str  # host, service, application, database
    
    # Connectivité
    endpoint: str
    port: Optional[int] = None
    protocol: str = "http"
    
    # Authentification
    auth_required: bool = False
    auth_username: str = ""
    auth_password: str = ""
    auth_token: str = ""
    
    # Configuration monitoring
    check_interval: int = 60
    timeout: int = 30
    retries: int = 3
    
    # Health checks
    health_endpoint: str = "/health"
    expected_status_code: int = 200
    expected_response_time_ms: int = 1000
    
    # Métriques spécifiques
    custom_metrics: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # État
    enabled: bool = True
    last_check: Optional[datetime] = None
    status: str = "unknown"


class AlertEngine:
    """Moteur d'évaluation d'alertes intelligent."""
    
    def __init__(self, metrics_system: EnterpriseMetricsSystem):
        self.metrics_system = metrics_system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # État du moteur
        self.running = False
        self.evaluation_interval = 30  # secondes
        
        # Cache et optimisations
        self.metric_cache: Dict[str, List[MetricDataPoint]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_cleanup = time.time()
        
        # Statistiques
        self.evaluations_count = 0
        self.alerts_triggered = 0
        self.alerts_resolved = 0
        
        # Machine Learning
        self.anomaly_models: Dict[str, Any] = {}
        self.baseline_data: Dict[str, List[float]] = {}
        
    async def start(self):
        """Démarre le moteur d'alertes."""
        if self.running:
            return
            
        self.running = True
        logging.info("🚨 Démarrage du moteur d'alertes")
        
        # Chargement des règles par défaut
        await self._load_default_rules()
        
        # Démarrage de la boucle d'évaluation
        asyncio.create_task(self._evaluation_loop())
        
        # Démarrage des tâches de maintenance
        asyncio.create_task(self._maintenance_loop())
        
    async def stop(self):
        """Arrête le moteur d'alertes."""
        self.running = False
        logging.info("🛑 Arrêt du moteur d'alertes")
        
    async def add_rule(self, rule: AlertRule):
        """Ajoute une règle d'alerte."""
        self.alert_rules[rule.rule_id] = rule
        
        # Initialisation du modèle ML si nécessaire
        if rule.use_anomaly_detection:
            await self._initialize_anomaly_model(rule)
        
        logging.info(f"📋 Règle d'alerte ajoutée: {rule.name}")
        
    async def remove_rule(self, rule_id: str):
        """Supprime une règle d'alerte."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logging.info(f"🗑️ Règle d'alerte supprimée: {rule_id}")
            
    async def acknowledge_alert(self, alert_id: str, user: str, notes: str = ""):
        """Acquitte une alerte."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledgments.append({
                "user": user,
                "timestamp": datetime.now().isoformat(),
                "notes": notes
            })
            
            logging.info(f"✅ Alerte acquittée: {alert_id} par {user}")
            
    async def resolve_alert(self, alert_id: str, user: str, resolution_notes: str = "", root_cause: str = ""):
        """Résout une alerte."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            alert.root_cause = root_cause
            
            # Déplacement vers l'historique
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.alerts_resolved += 1
            
            logging.info(f"✅ Alerte résolue: {alert_id} par {user}")
            
    async def _evaluation_loop(self):
        """Boucle principale d'évaluation des alertes."""
        while self.running:
            try:
                start_time = time.time()
                
                # Évaluation de toutes les règles actives
                for rule_id, rule in self.alert_rules.items():
                    if rule.enabled:
                        await self._evaluate_rule(rule)
                
                # Auto-résolution des alertes
                await self._auto_resolve_alerts()
                
                # Nettoyage du cache
                await self._cleanup_cache()
                
                self.evaluations_count += 1
                
                evaluation_time = time.time() - start_time
                logging.debug(f"🔍 Évaluation complétée en {evaluation_time:.2f}s")
                
                # Attente avant prochaine évaluation
                await asyncio.sleep(max(0, self.evaluation_interval - evaluation_time))
                
            except Exception as e:
                logging.error(f"❌ Erreur dans l'évaluation des alertes: {e}")
                await asyncio.sleep(self.evaluation_interval)
                
    async def _evaluate_rule(self, rule: AlertRule):
        """Évalue une règle d'alerte spécifique."""
        try:
            # Récupération des métriques
            metrics = await self._get_metrics_for_rule(rule)
            
            if not metrics:
                return
            
            # Évaluation selon le type de règle
            if rule.use_anomaly_detection:
                triggered = await self._evaluate_anomaly_rule(rule, metrics)
            else:
                triggered = await self._evaluate_threshold_rule(rule, metrics)
            
            if triggered:
                await self._trigger_alert(rule, metrics[-1])  # Dernière métrique
                
        except Exception as e:
            logging.error(f"❌ Erreur évaluation règle {rule.rule_id}: {e}")
            
    async def _get_metrics_for_rule(self, rule: AlertRule) -> List[MetricDataPoint]:
        """Récupère les métriques pour une règle."""
        cache_key = f"{rule.rule_id}_{rule.metric_pattern}"
        current_time = time.time()
        
        # Vérification du cache
        if cache_key in self.metric_cache:
            cached_data, cache_time = self.metric_cache[cache_key]
            if current_time - cache_time < self.cache_ttl:
                return cached_data
        
        # Récupération depuis le stockage
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=rule.duration_seconds + 300)  # +5min buffer
        
        try:
            # Utilisation d'un pattern regex pour matcher les métriques
            import re
            pattern = re.compile(rule.metric_pattern)
            
            # Récupération de toutes les métriques récentes
            all_metrics = await self.metrics_system.storage.query_metrics(
                start_time=start_time,
                end_time=end_time,
                category=rule.category
            )
            
            # Filtrage par pattern
            filtered_metrics = [
                metric for metric in all_metrics 
                if pattern.match(metric.metric_id)
            ]
            
            # Mise en cache
            self.metric_cache[cache_key] = (filtered_metrics, current_time)
            
            return filtered_metrics
            
        except Exception as e:
            logging.error(f"❌ Erreur récupération métriques pour règle {rule.rule_id}: {e}")
            return []
            
    async def _evaluate_threshold_rule(self, rule: AlertRule, metrics: List[MetricDataPoint]) -> bool:
        """Évalue une règle basée sur des seuils."""
        if not metrics:
            return False
        
        # Filtrage des métriques dans la fenêtre de durée
        cutoff_time = datetime.now() - timedelta(seconds=rule.duration_seconds)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return False
        
        # Évaluation de la condition
        for metric in recent_metrics:
            if not self._check_threshold_condition(metric.value, rule.threshold_value, rule.comparison):
                return False  # Une métrique ne respecte pas la condition
        
        # Toutes les métriques dans la fenêtre respectent la condition
        return len(recent_metrics) >= (rule.duration_seconds // 60)  # Au moins 1 métrique par minute
        
    async def _evaluate_anomaly_rule(self, rule: AlertRule, metrics: List[MetricDataPoint]) -> bool:
        """Évalue une règle basée sur la détection d'anomalies."""
        if not metrics or not HAS_SCIPY:
            return False
        
        # Préparation des données pour l'analyse
        values = [m.value for m in metrics]
        
        if len(values) < 10:  # Pas assez de données
            return False
        
        # Calcul des statistiques
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0
        
        # Dernière valeur
        last_value = values[-1]
        
        # Détection d'anomalie simple (z-score)
        if std_value > 0:
            z_score = abs(last_value - mean_value) / std_value
            threshold = 3.0 * rule.ml_sensitivity  # Seuil ajustable
            
            return z_score > threshold
        
        return False
        
    def _check_threshold_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Vérifie une condition de seuil."""
        try:
            if comparison == ">":
                return value > threshold
            elif comparison == "<":
                return value < threshold
            elif comparison == ">=":
                return value >= threshold
            elif comparison == "<=":
                return value <= threshold
            elif comparison == "==":
                return abs(value - threshold) < 1e-9  # Égalité avec tolérance
            elif comparison == "!=":
                return abs(value - threshold) >= 1e-9
            else:
                logging.warning(f"⚠️ Comparaison inconnue: {comparison}")
                return False
        except Exception:
            return False
            
    async def _trigger_alert(self, rule: AlertRule, metric: MetricDataPoint):
        """Déclenche une alerte."""
        # Vérification de suppression
        suppression_key = f"{rule.rule_id}_{metric.metric_id}"
        if await self._is_suppressed(suppression_key):
            return
        
        # Génération de l'ID d'alerte
        alert_id = f"alert_{rule.rule_id}_{int(time.time())}"
        
        # Création de l'alerte
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            title=f"Alerte: {rule.name}",
            description=rule.description,
            metric_name=metric.metric_id,
            current_value=metric.value,
            threshold_value=rule.threshold_value,
            priority=rule.priority,
            category=rule.category,
            tags=rule.tags.copy(),
            metadata={
                "rule_name": rule.name,
                "metric_source": metric.source,
                "metric_tags": metric.tags,
                "comparison": rule.comparison
            }
        )
        
        # Ajout à la liste des alertes actives
        self.active_alerts[alert_id] = alert
        self.alerts_triggered += 1
        
        logging.warning(f"🚨 Alerte déclenchée: {rule.name} - {metric.metric_id} = {metric.value}")
        
        # Stockage de l'alerte
        await self._store_alert(alert)
        
        # Envoi des notifications
        asyncio.create_task(self._send_notifications(alert, rule))
        
    async def _auto_resolve_alerts(self):
        """Auto-résolution des alertes."""
        current_time = datetime.now()
        
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.auto_resolve:
                continue
            
            # Vérification du délai d'auto-résolution
            time_since_trigger = (current_time - alert.triggered_at).total_seconds()
            
            if time_since_trigger > rule.auto_resolve_duration:
                # Vérification si la condition n'est plus vraie
                if await self._is_condition_resolved(alert, rule):
                    alerts_to_resolve.append(alert_id)
        
        # Résolution automatique
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id, "system", "Auto-résolution - condition non respectée")
            
    async def _is_condition_resolved(self, alert: Alert, rule: AlertRule) -> bool:
        """Vérifie si la condition d'une alerte est résolue."""
        try:
            # Récupération de la métrique actuelle
            metrics = await self._get_metrics_for_rule(rule)
            
            if not metrics:
                return True  # Pas de données = considéré comme résolu
            
            # Vérification de la condition sur les dernières métriques
            recent_metrics = metrics[-5:]  # 5 dernières métriques
            
            for metric in recent_metrics:
                if self._check_threshold_condition(metric.value, rule.threshold_value, rule.comparison):
                    return False  # Condition encore vraie
            
            return True  # Condition résolue
            
        except Exception as e:
            logging.error(f"❌ Erreur vérification résolution pour {alert.alert_id}: {e}")
            return False
            
    async def _maintenance_loop(self):
        """Boucle de maintenance du moteur."""
        while self.running:
            try:
                # Nettoyage de l'historique ancien
                await self._cleanup_old_alerts()
                
                # Mise à jour des modèles ML
                await self._update_ml_models()
                
                # Statistiques et métriques du moteur
                await self._publish_engine_metrics()
                
                # Maintenance toutes les heures
                await asyncio.sleep(3600)
                
            except Exception as e:
                logging.error(f"❌ Erreur maintenance moteur: {e}")
                await asyncio.sleep(3600)
                
    async def _cleanup_old_alerts(self):
        """Nettoie les anciennes alertes."""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Nettoyage de l'historique
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
        
    async def _publish_engine_metrics(self):
        """Publie les métriques du moteur."""
        timestamp = datetime.now()
        
        engine_metrics = [
            MetricDataPoint(
                metric_id="alert_engine.evaluations_total",
                timestamp=timestamp,
                value=self.evaluations_count,
                metric_type=MetricType.CUMULATIVE,
                category=MetricCategory.SYSTEM,
                tags={"component": "alert_engine"}
            ),
            MetricDataPoint(
                metric_id="alert_engine.alerts_active",
                timestamp=timestamp,
                value=len(self.active_alerts),
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                tags={"component": "alert_engine"}
            ),
            MetricDataPoint(
                metric_id="alert_engine.alerts_triggered_total",
                timestamp=timestamp,
                value=self.alerts_triggered,
                metric_type=MetricType.CUMULATIVE,
                category=MetricCategory.SYSTEM,
                tags={"component": "alert_engine"}
            ),
            MetricDataPoint(
                metric_id="alert_engine.alerts_resolved_total",
                timestamp=timestamp,
                value=self.alerts_resolved,
                metric_type=MetricType.CUMULATIVE,
                category=MetricCategory.SYSTEM,
                tags={"component": "alert_engine"}
            )
        ]
        
        # Stockage des métriques
        for metric in engine_metrics:
            await self.metrics_system.storage.store_metric(metric)
            
    async def _load_default_rules(self):
        """Charge les règles d'alerte par défaut."""
        default_rules = [
            # Règles système
            AlertRule(
                rule_id="cpu_high",
                name="CPU Usage High",
                description="CPU usage above 90% for 5 minutes",
                metric_pattern=r"system\.cpu\.usage_total",
                threshold_value=90.0,
                comparison=">",
                duration_seconds=300,
                priority=AlertPriority.HIGH,
                category=MetricCategory.SYSTEM,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                rule_id="memory_high",
                name="Memory Usage High",
                description="Memory usage above 85% for 5 minutes",
                metric_pattern=r"system\.memory\.percent",
                threshold_value=85.0,
                comparison=">",
                duration_seconds=300,
                priority=AlertPriority.HIGH,
                category=MetricCategory.SYSTEM,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            AlertRule(
                rule_id="disk_space_low",
                name="Disk Space Low",
                description="Disk space above 80% usage",
                metric_pattern=r"system\.disk\.percent",
                threshold_value=80.0,
                comparison=">",
                duration_seconds=60,
                priority=AlertPriority.MEDIUM,
                category=MetricCategory.STORAGE,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            # Règles de sécurité
            AlertRule(
                rule_id="auth_failures_high",
                name="Authentication Failures High",
                description="High number of authentication failures",
                metric_pattern=r"security\.authentication\.failed_attempts",
                threshold_value=10.0,
                comparison=">",
                duration_seconds=300,
                priority=AlertPriority.CRITICAL,
                category=MetricCategory.SECURITY,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            # Règles de performance
            AlertRule(
                rule_id="response_time_high",
                name="API Response Time High",
                description="API response time above 1000ms",
                metric_pattern=r"application\.api\.response_time",
                threshold_value=1000.0,
                comparison=">",
                duration_seconds=120,
                priority=AlertPriority.MEDIUM,
                category=MetricCategory.PERFORMANCE,
                notification_channels=[NotificationChannel.EMAIL]
            ),
            # Règles réseau
            AlertRule(
                rule_id="network_errors_high",
                name="Network Errors High",
                description="High network error rate",
                metric_pattern=r"system\.network\.(errin|errout)",
                threshold_value=100.0,
                comparison=">",
                duration_seconds=180,
                priority=AlertPriority.HIGH,
                category=MetricCategory.NETWORK,
                notification_channels=[NotificationChannel.EMAIL]
            )
        ]
        
        for rule in default_rules:
            await self.add_rule(rule)
            
        logging.info(f"📋 {len(default_rules)} règles par défaut chargées")
        
    async def _is_suppressed(self, suppression_key: str) -> bool:
        """Vérifie si une alerte est supprimée."""
        # Implémentation simple - en production, utiliser Redis ou DB
        return False
        
    async def _store_alert(self, alert: Alert):
        """Stocke une alerte."""
        # Stockage comme métrique pour le tracking
        metric = MetricDataPoint(
            metric_id="alert_engine.alert_triggered",
            timestamp=alert.triggered_at,
            value=1.0,
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SYSTEM,
            tags={
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "priority": alert.priority.value,
                "metric_name": alert.metric_name
            },
            metadata={
                "alert_title": alert.title,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value
            }
        )
        
        await self.metrics_system.storage.store_metric(metric)
        
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Envoie les notifications pour une alerte."""
        for channel in rule.notification_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert, rule)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert, rule)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert, rule)
                    
                alert.notifications_sent.append(f"{channel.value}:{datetime.now().isoformat()}")
                
            except Exception as e:
                logging.error(f"❌ Erreur envoi notification {channel.value}: {e}")
                
    async def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Envoie une notification par email."""
        # Configuration email - à adapter selon l'environnement
        smtp_server = os.getenv("SMTP_SERVER", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        if not smtp_server:
            logging.warning("⚠️ Configuration SMTP manquante")
            return
            
        # Création du message
        subject = f"🚨 Alert: {alert.title}"
        
        body = f"""
Alert Details:
=============

Alert ID: {alert.alert_id}
Rule: {rule.name}
Priority: {alert.priority.value.upper()}
Status: {alert.status.value}

Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}

Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}

Description:
{alert.description}

Tags: {', '.join([f'{k}={v}' for k, v in alert.tags.items()])}

---
Enterprise Metrics System
"""
        
        # Envoi de l'email
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_username
            msg['To'] = ", ".join(["admin@example.com"])  # À configurer
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            # Connexion SMTP
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                server.send_message(msg)
                
            logging.info(f"📧 Email envoyé pour alerte {alert.alert_id}")
            
        except Exception as e:
            logging.error(f"❌ Erreur envoi email: {e}")
            
    async def _send_slack_notification(self, alert: Alert, rule: AlertRule):
        """Envoie une notification Slack."""
        if not HAS_SLACK:
            logging.warning("⚠️ Module Slack non disponible")
            return
            
        # À implémenter avec le webhook Slack
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        
        if not slack_webhook:
            logging.warning("⚠️ URL webhook Slack manquante")
            return
            
        # Formatage du message Slack
        color = {
            AlertPriority.EMERGENCY: "danger",
            AlertPriority.CRITICAL: "danger", 
            AlertPriority.HIGH: "warning",
            AlertPriority.MEDIUM: "warning",
            AlertPriority.LOW: "good",
            AlertPriority.INFO: "good"
        }.get(alert.priority, "warning")
        
        message = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.title}",
                "text": alert.description,
                "fields": [
                    {"title": "Priority", "value": alert.priority.value.upper(), "short": True},
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                    {"title": "Rule", "value": rule.name, "short": True},
                    {"title": "Alert ID", "value": alert.alert_id, "short": True}
                ],
                "ts": int(alert.triggered_at.timestamp())
            }]
        }
        
        # Envoi via webhook
        if HAS_REQUESTS:
            try:
                response = requests.post(slack_webhook, json=message, timeout=10)
                response.raise_for_status()
                logging.info(f"📱 Notification Slack envoyée pour {alert.alert_id}")
            except Exception as e:
                logging.error(f"❌ Erreur envoi Slack: {e}")
                
    async def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Envoie une notification via webhook."""
        webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
        
        if not webhook_url:
            return
            
        # Payload du webhook
        payload = {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "title": alert.title,
            "description": alert.description,
            "priority": alert.priority.value,
            "status": alert.status.value,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "triggered_at": alert.triggered_at.isoformat(),
            "tags": alert.tags,
            "metadata": alert.metadata
        }
        
        # Envoi du webhook
        if HAS_REQUESTS:
            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                logging.info(f"🔗 Webhook envoyé pour {alert.alert_id}")
            except Exception as e:
                logging.error(f"❌ Erreur envoi webhook: {e}")
                
    async def _cleanup_cache(self):
        """Nettoie le cache des métriques."""
        current_time = time.time()
        
        if current_time - self.last_cache_cleanup > 300:  # Nettoyage toutes les 5 minutes
            expired_keys = []
            
            for key, (data, cache_time) in self.metric_cache.items():
                if current_time - cache_time > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.metric_cache[key]
                
            self.last_cache_cleanup = current_time
            
            if expired_keys:
                logging.debug(f"🧹 Cache nettoyé: {len(expired_keys)} entrées supprimées")
                
    async def _initialize_anomaly_model(self, rule: AlertRule):
        """Initialise un modèle de détection d'anomalies."""
        # Implémentation basique - en production, utiliser des modèles plus sophistiqués
        self.anomaly_models[rule.rule_id] = {
            "type": "zscore",
            "sensitivity": rule.ml_sensitivity,
            "baseline_window": rule.baseline_window_hours
        }
        
    async def _update_ml_models(self):
        """Met à jour les modèles ML."""
        # Mise à jour périodique des modèles d'anomalie
        for rule_id, model in self.anomaly_models.items():
            try:
                # Récupération des données d'entraînement
                rule = self.alert_rules.get(rule_id)
                if rule:
                    # Mise à jour du baseline
                    await self._update_baseline_data(rule)
            except Exception as e:
                logging.error(f"❌ Erreur mise à jour modèle {rule_id}: {e}")
                
    async def _update_baseline_data(self, rule: AlertRule):
        """Met à jour les données de baseline."""
        # Récupération des données historiques
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=rule.baseline_window_hours)
        
        metrics = await self._get_metrics_for_rule(rule)
        baseline_data = [m.value for m in metrics if start_time <= m.timestamp <= end_time]
        
        if len(baseline_data) >= 10:  # Minimum de données
            self.baseline_data[rule.rule_id] = baseline_data
            
    def get_engine_status(self) -> Dict[str, Any]:
        """Retourne le statut du moteur d'alertes."""
        return {
            "running": self.running,
            "rules_count": len(self.alert_rules),
            "active_alerts_count": len(self.active_alerts),
            "total_evaluations": self.evaluations_count,
            "alerts_triggered": self.alerts_triggered,
            "alerts_resolved": self.alerts_resolved,
            "cache_entries": len(self.metric_cache),
            "ml_models": len(self.anomaly_models)
        }
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne la liste des alertes actives."""
        return [
            {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "priority": alert.priority.value,
                "status": alert.status.value,
                "triggered_at": alert.triggered_at.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value
            }
            for alert in self.active_alerts.values()
        ]


class HealthMonitor:
    """Moniteur de santé des services."""
    
    def __init__(self, alert_engine: AlertEngine):
        self.alert_engine = alert_engine
        self.targets: Dict[str, MonitoringTarget] = {}
        self.running = False
        self.check_interval = 60
        
    async def start(self):
        """Démarre le moniteur de santé."""
        if self.running:
            return
            
        self.running = True
        logging.info("💓 Démarrage du moniteur de santé")
        
        # Chargement des cibles par défaut
        await self._load_default_targets()
        
        # Démarrage de la boucle de vérification
        asyncio.create_task(self._health_check_loop())
        
    async def stop(self):
        """Arrête le moniteur de santé."""
        self.running = False
        logging.info("🛑 Arrêt du moniteur de santé")
        
    async def add_target(self, target: MonitoringTarget):
        """Ajoute une cible de monitoring."""
        self.targets[target.target_id] = target
        logging.info(f"🎯 Cible ajoutée: {target.name}")
        
    async def remove_target(self, target_id: str):
        """Supprime une cible de monitoring."""
        if target_id in self.targets:
            del self.targets[target_id]
            logging.info(f"🗑️ Cible supprimée: {target_id}")
            
    async def _health_check_loop(self):
        """Boucle de vérification de santé."""
        while self.running:
            try:
                # Vérification de toutes les cibles actives
                for target_id, target in self.targets.items():
                    if target.enabled:
                        asyncio.create_task(self._check_target_health(target))
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"❌ Erreur boucle santé: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _check_target_health(self, target: MonitoringTarget):
        """Vérifie la santé d'une cible."""
        try:
            start_time = time.time()
            
            # Health check HTTP
            if target.protocol == "http" or target.protocol == "https":
                health_ok, response_time = await self._http_health_check(target)
            else:
                health_ok, response_time = await self._tcp_health_check(target)
            
            # Mise à jour du statut
            target.last_check = datetime.now()
            target.status = "healthy" if health_ok else "unhealthy"
            
            # Création de métriques de santé
            await self._publish_health_metrics(target, health_ok, response_time)
            
        except Exception as e:
            logging.error(f"❌ Erreur vérification santé {target.name}: {e}")
            target.status = "error"
            
    async def _http_health_check(self, target: MonitoringTarget) -> Tuple[bool, float]:
        """Effectue un health check HTTP."""
        if not HAS_REQUESTS:
            return False, 0.0
            
        try:
            url = f"{target.protocol}://{target.endpoint}:{target.port or 80}{target.health_endpoint}"
            
            start_time = time.time()
            
            # Préparation de l'authentification
            auth = None
            headers = {}
            
            if target.auth_required:
                if target.auth_token:
                    headers["Authorization"] = f"Bearer {target.auth_token}"
                elif target.auth_username and target.auth_password:
                    auth = (target.auth_username, target.auth_password)
            
            # Requête HTTP
            response = requests.get(
                url,
                timeout=target.timeout,
                auth=auth,
                headers=headers,
                verify=False  # En production, configurer correctement SSL
            )
            
            response_time = (time.time() - start_time) * 1000  # en ms
            
            # Vérification du statut et du temps de réponse
            status_ok = response.status_code == target.expected_status_code
            time_ok = response_time <= target.expected_response_time_ms
            
            return status_ok and time_ok, response_time
            
        except Exception as e:
            logging.debug(f"⚠️ Health check échoué pour {target.name}: {e}")
            return False, 0.0
            
    async def _tcp_health_check(self, target: MonitoringTarget) -> Tuple[bool, float]:
        """Effectue un health check TCP."""
        try:
            start_time = time.time()
            
            # Connexion TCP
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(target.timeout)
                result = sock.connect_ex((target.endpoint, target.port or 80))
                
            response_time = (time.time() - start_time) * 1000  # en ms
            
            return result == 0, response_time
            
        except Exception as e:
            logging.debug(f"⚠️ TCP check échoué pour {target.name}: {e}")
            return False, 0.0
            
    async def _publish_health_metrics(self, target: MonitoringTarget, health_ok: bool, response_time: float):
        """Publie les métriques de santé."""
        timestamp = datetime.now()
        
        metrics = [
            MetricDataPoint(
                metric_id="health_monitor.target_status",
                timestamp=timestamp,
                value=1.0 if health_ok else 0.0,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.AVAILABILITY,
                tags={
                    "target_id": target.target_id,
                    "target_name": target.name,
                    "target_type": target.target_type,
                    "protocol": target.protocol
                }
            ),
            MetricDataPoint(
                metric_id="health_monitor.response_time",
                timestamp=timestamp,
                value=response_time,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                tags={
                    "target_id": target.target_id,
                    "target_name": target.name,
                    "target_type": target.target_type
                },
                metadata={"unit": "milliseconds"}
            )
        ]
        
        # Stockage des métriques
        for metric in metrics:
            await self.alert_engine.metrics_system.storage.store_metric(metric)
            
    async def _load_default_targets(self):
        """Charge les cibles de monitoring par défaut."""
        default_targets = [
            MonitoringTarget(
                target_id="localhost_api",
                name="Local API",
                target_type="api",
                endpoint="127.0.0.1",
                port=8080,
                protocol="http",
                health_endpoint="/health",
                check_interval=30
            ),
            MonitoringTarget(
                target_id="localhost_metrics",
                name="Metrics Endpoint",
                target_type="metrics",
                endpoint="127.0.0.1",
                port=9090,
                protocol="http",
                health_endpoint="/metrics",
                check_interval=60
            )
        ]
        
        for target in default_targets:
            await self.add_target(target)
            
        logging.info(f"🎯 {len(default_targets)} cibles par défaut chargées")


async def main():
    """Fonction principale du système de monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Système de monitoring et d'alertes")
    parser.add_argument("--storage", default="sqlite", help="Type de stockage")
    parser.add_argument("--config", help="Fichier de configuration")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialisation du système de métriques
        metrics_system = get_metrics_system(args.storage)
        await metrics_system.start()
        
        # Initialisation du moteur d'alertes
        alert_engine = AlertEngine(metrics_system)
        await alert_engine.start()
        
        # Initialisation du moniteur de santé
        health_monitor = HealthMonitor(alert_engine)
        await health_monitor.start()
        
        logging.info("🚀 Système de monitoring démarré")
        
        # Boucle principale
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("⏹️ Arrêt du système de monitoring")
            
        # Arrêt propre
        await health_monitor.stop()
        await alert_engine.stop()
        await metrics_system.stop()
        
        logging.info("✅ Système de monitoring arrêté")
        
    except Exception as e:
        logging.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
