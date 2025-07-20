#!/usr/bin/env python3
"""
Monitoring Automation Scripts - Scripts d'Automatisation du Monitoring
=====================================================================

Scripts d'automatisation pour la collecte, l'analyse et l'alerting
du système de monitoring Spotify AI Agent.

Features:
    - Collecte automatisée des métriques
    - Génération de rapports planifiés
    - Alerting intelligent et adaptatif
    - Maintenance automatique du système
    - Optimisation des performances

Author: Expert DevOps Automation + Site Reliability Engineering Team
"""

import asyncio
import sys
import os
import json
import yaml
import logging
import argparse
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import signal
import traceback
from dataclasses import dataclass, field
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
import redis
import psutil
import subprocess

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION ET UTILITAIRES
# =============================================================================

@dataclass
class AutomationConfig:
    """Configuration pour l'automatisation."""
    collection_interval_seconds: int = 60
    report_generation_schedule: str = "daily"
    alert_cooldown_minutes: int = 15
    max_workers: int = 4
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///monitoring.db"
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    debug_mode: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AutomationConfig':
        """Charge la configuration depuis un fichier."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return cls()


class MonitoringDatabase:
    """Interface base de données pour le monitoring."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._setup_tables()
    
    def _setup_tables(self):
        """Crée les tables nécessaires."""
        conn = sqlite3.connect(self.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        # Table des métriques
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                metadata TEXT
            )
        ''')
        
        # Table des alertes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                tenant_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME,
                is_active BOOLEAN DEFAULT 1,
                metadata TEXT
            )
        ''')
        
        # Table des rapports
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                tenant_id TEXT NOT NULL,
                report_type TEXT NOT NULL,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                metadata TEXT
            )
        ''')
        
        # Index pour performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_tenant_time ON metrics(tenant_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_tenant_active ON alerts(tenant_id, is_active)')
        
        conn.commit()
        conn.close()
    
    def save_metric(self, tenant_id: str, metric_name: str, value: float, 
                   metric_type: str, tags: Dict[str, str] = None, 
                   metadata: Dict[str, Any] = None):
        """Sauvegarde une métrique."""
        conn = sqlite3.connect(self.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (tenant_id, metric_name, metric_value, metric_type, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            tenant_id, metric_name, value, metric_type,
            json.dumps(tags or {}), json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert_data: Dict[str, Any]):
        """Sauvegarde une alerte."""
        conn = sqlite3.connect(self.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (alert_id, tenant_id, severity, title, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert_data['alert_id'], alert_data['tenant_id'],
            alert_data['severity'], alert_data['title'],
            alert_data.get('description', ''), json.dumps(alert_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_alerts(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """Récupère les alertes actives."""
        conn = sqlite3.connect(self.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        query = 'SELECT * FROM alerts WHERE is_active = 1'
        params = []
        
        if tenant_id:
            query += ' AND tenant_id = ?'
            params.append(tenant_id)
        
        cursor.execute(query, params)
        alerts = []
        
        for row in cursor.fetchall():
            alert = {
                'id': row[0],
                'alert_id': row[1],
                'tenant_id': row[2],
                'severity': row[3],
                'title': row[4],
                'description': row[5],
                'triggered_at': row[6],
                'resolved_at': row[7],
                'is_active': row[8],
                'metadata': json.loads(row[9]) if row[9] else {}
            }
            alerts.append(alert)
        
        conn.close()
        return alerts


class NotificationManager:
    """Gestionnaire de notifications."""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.channels = config.notification_channels
    
    async def send_alert(self, alert_data: Dict[str, Any]):
        """Envoie une alerte via tous les canaux configurés."""
        tasks = []
        
        if "email" in self.channels:
            tasks.append(self._send_email_alert(alert_data))
        
        if "slack" in self.channels:
            tasks.append(self._send_slack_alert(alert_data))
        
        if "webhook" in self.channels:
            tasks.append(self._send_webhook_alert(alert_data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Envoie une alerte par email."""
        try:
            # Configuration email (à adapter selon votre setup)
            smtp_server = os.getenv('SMTP_SERVER', 'localhost')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER', '')
            smtp_password = os.getenv('SMTP_PASSWORD', '')
            
            msg = MimeMultipart()
            msg['From'] = smtp_user
            msg['To'] = os.getenv('ALERT_EMAIL', 'admin@example.com')
            msg['Subject'] = f"[{alert_data['severity'].upper()}] {alert_data['title']}"
            
            body = f"""
Alerte Monitoring Spotify AI Agent

Tenant: {alert_data['tenant_id']}
Sévérité: {alert_data['severity']}
Titre: {alert_data['title']}
Description: {alert_data.get('description', 'N/A')}
Déclenché à: {alert_data.get('triggered_at', datetime.utcnow())}

Métadonnées:
{json.dumps(alert_data.get('metadata', {}), indent=2)}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Alerte email envoyée: {alert_data['alert_id']}")
            
        except Exception as e:
            logger.error(f"Erreur envoi email: {e}")
    
    async def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Envoie une alerte sur Slack."""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                logger.warning("Slack webhook URL non configurée")
                return
            
            # Couleur selon la sévérité
            color_map = {
                'critical': '#FF0000',
                'high': '#FF8C00',
                'medium': '#FFD700',
                'low': '#90EE90',
                'info': '#87CEEB'
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert_data['severity'], '#808080'),
                        "title": f"🚨 {alert_data['title']}",
                        "fields": [
                            {
                                "title": "Tenant",
                                "value": alert_data['tenant_id'],
                                "short": True
                            },
                            {
                                "title": "Sévérité",
                                "value": alert_data['severity'].upper(),
                                "short": True
                            },
                            {
                                "title": "Description",
                                "value": alert_data.get('description', 'N/A'),
                                "short": False
                            }
                        ],
                        "footer": "Spotify AI Agent Monitoring",
                        "ts": int(time.time())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Alerte Slack envoyée: {alert_data['alert_id']}")
            
        except Exception as e:
            logger.error(f"Erreur envoi Slack: {e}")
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Envoie une alerte via webhook."""
        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            if not webhook_url:
                return
            
            payload = {
                "event_type": "alert",
                "alert_data": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Alerte webhook envoyée: {alert_data['alert_id']}")
            
        except Exception as e:
            logger.error(f"Erreur envoi webhook: {e}")


# =============================================================================
# COLLECTEURS AUTOMATISÉS
# =============================================================================

class AutomatedCollector:
    """Collecteur automatisé de métriques."""
    
    def __init__(self, config: AutomationConfig, db: MonitoringDatabase):
        self.config = config
        self.db = db
        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        self.is_running = False
        self.collection_tasks = []
    
    async def start_collection(self):
        """Démarre la collecte automatisée."""
        self.is_running = True
        logger.info("Démarrage de la collecte automatisée")
        
        # Planification des tâches
        schedule.every(self.config.collection_interval_seconds).seconds.do(
            self._schedule_collection
        )
        
        # Boucle principale
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(1)
    
    def stop_collection(self):
        """Arrête la collecte."""
        self.is_running = False
        logger.info("Arrêt de la collecte automatisée")
    
    def _schedule_collection(self):
        """Planifie une collecte."""
        if not self.is_running:
            return
        
        # Créer une tâche asynchrone pour la collecte
        task = asyncio.create_task(self._run_collection_cycle())
        self.collection_tasks.append(task)
        
        # Nettoyer les tâches terminées
        self.collection_tasks = [t for t in self.collection_tasks if not t.done()]
    
    async def _run_collection_cycle(self):
        """Exécute un cycle de collecte complet."""
        try:
            start_time = time.time()
            logger.info("Début du cycle de collecte")
            
            # Collecte des métriques système
            await self._collect_system_metrics()
            
            # Collecte des métriques applicatives
            await self._collect_application_metrics()
            
            # Collecte des métriques business
            await self._collect_business_metrics()
            
            # Collecte des métriques de sécurité
            await self._collect_security_metrics()
            
            duration = time.time() - start_time
            logger.info(f"Cycle de collecte terminé en {duration:.2f}s")
            
            # Stocker la métrique de performance de collecte
            self.db.save_metric(
                tenant_id="system",
                metric_name="collection_duration_seconds",
                value=duration,
                metric_type="gauge"
            )
            
        except Exception as e:
            logger.error(f"Erreur dans le cycle de collecte: {e}")
            traceback.print_exc()
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.db.save_metric("system", "cpu_usage_percent", cpu_percent, "gauge")
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.db.save_metric("system", "memory_usage_percent", memory.percent, "gauge")
            self.db.save_metric("system", "memory_used_bytes", memory.used, "gauge")
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.db.save_metric("system", "disk_usage_percent", disk_percent, "gauge")
            
            # Réseau
            network = psutil.net_io_counters()
            self.db.save_metric("system", "network_bytes_sent", network.bytes_sent, "counter")
            self.db.save_metric("system", "network_bytes_recv", network.bytes_recv, "counter")
            
            # Processus
            process_count = len(psutil.pids())
            self.db.save_metric("system", "process_count", process_count, "gauge")
            
            logger.debug("Métriques système collectées")
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
    
    async def _collect_application_metrics(self):
        """Collecte les métriques applicatives."""
        try:
            # Simuler des métriques applicatives
            # En production, ces métriques viendraient des vrais collecteurs
            
            tenants = ["tenant_001", "tenant_002", "tenant_003"]
            
            for tenant_id in tenants:
                # Métriques API
                api_response_time = 100 + (time.time() % 100)  # Simulation
                self.db.save_metric(tenant_id, "api_response_time_ms", api_response_time, "gauge")
                
                # Métriques utilisateurs
                active_users = 50 + int(time.time() % 200)  # Simulation
                self.db.save_metric(tenant_id, "active_users", active_users, "gauge")
                
                # Métriques erreurs
                error_rate = 0.01 + (time.time() % 0.05)  # Simulation
                self.db.save_metric(tenant_id, "error_rate", error_rate, "gauge")
            
            logger.debug("Métriques applicatives collectées")
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques applicatives: {e}")
    
    async def _collect_business_metrics(self):
        """Collecte les métriques business."""
        try:
            # Simuler des métriques business
            revenue = 10000 + (time.time() % 5000)  # Simulation
            self.db.save_metric("business", "daily_revenue", revenue, "gauge")
            
            conversion_rate = 0.15 + (time.time() % 0.1)  # Simulation
            self.db.save_metric("business", "conversion_rate", conversion_rate, "gauge")
            
            churn_rate = 0.05 + (time.time() % 0.03)  # Simulation
            self.db.save_metric("business", "churn_rate", churn_rate, "gauge")
            
            logger.debug("Métriques business collectées")
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques business: {e}")
    
    async def _collect_security_metrics(self):
        """Collecte les métriques de sécurité."""
        try:
            # Simuler des métriques de sécurité
            failed_logins = int(time.time() % 20)  # Simulation
            self.db.save_metric("security", "failed_login_attempts", failed_logins, "counter")
            
            blocked_ips = int(time.time() % 5)  # Simulation
            self.db.save_metric("security", "blocked_ips_count", blocked_ips, "gauge")
            
            security_score = 85 + (time.time() % 15)  # Simulation
            self.db.save_metric("security", "security_score", security_score, "gauge")
            
            logger.debug("Métriques de sécurité collectées")
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques sécurité: {e}")


# =============================================================================
# GESTIONNAIRE D'ALERTES
# =============================================================================

class AlertManager:
    """Gestionnaire d'alertes intelligent."""
    
    def __init__(self, config: AutomationConfig, db: MonitoringDatabase, 
                 notification_manager: NotificationManager):
        self.config = config
        self.db = db
        self.notification_manager = notification_manager
        self.alert_rules = self._load_alert_rules()
        self.alert_cooldowns = {}
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Charge les règles d'alerte."""
        return [
            {
                "name": "high_cpu_usage",
                "metric": "cpu_usage_percent",
                "threshold": 80,
                "operator": ">",
                "severity": "high",
                "duration_minutes": 5,
                "cooldown_minutes": 15
            },
            {
                "name": "high_memory_usage",
                "metric": "memory_usage_percent",
                "threshold": 85,
                "operator": ">",
                "severity": "high",
                "duration_minutes": 3,
                "cooldown_minutes": 10
            },
            {
                "name": "high_error_rate",
                "metric": "error_rate",
                "threshold": 0.1,
                "operator": ">",
                "severity": "critical",
                "duration_minutes": 2,
                "cooldown_minutes": 5
            },
            {
                "name": "low_conversion_rate",
                "metric": "conversion_rate",
                "threshold": 0.05,
                "operator": "<",
                "severity": "medium",
                "duration_minutes": 30,
                "cooldown_minutes": 60
            }
        ]
    
    async def check_alerts(self):
        """Vérifie et déclenche les alertes si nécessaire."""
        try:
            for rule in self.alert_rules:
                await self._check_rule(rule)
            
            # Nettoyer les cooldowns expirés
            self._cleanup_cooldowns()
            
        except Exception as e:
            logger.error(f"Erreur vérification alertes: {e}")
    
    async def _check_rule(self, rule: Dict[str, Any]):
        """Vérifie une règle d'alerte spécifique."""
        try:
            rule_name = rule["name"]
            
            # Vérifier si la règle est en cooldown
            if self._is_in_cooldown(rule_name):
                return
            
            # Récupérer les métriques récentes
            metrics = self._get_recent_metrics(
                rule["metric"], 
                rule.get("duration_minutes", 5)
            )
            
            if not metrics:
                return
            
            # Vérifier si la condition est remplie
            if self._evaluate_condition(metrics, rule):
                await self._trigger_alert(rule, metrics[-1])
        
        except Exception as e:
            logger.error(f"Erreur vérification règle {rule.get('name', 'unknown')}: {e}")
    
    def _get_recent_metrics(self, metric_name: str, duration_minutes: int) -> List[float]:
        """Récupère les métriques récentes."""
        conn = sqlite3.connect(self.config.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        since = datetime.utcnow() - timedelta(minutes=duration_minutes)
        
        cursor.execute('''
            SELECT metric_value FROM metrics 
            WHERE metric_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (metric_name, since.isoformat()))
        
        metrics = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return metrics
    
    def _evaluate_condition(self, metrics: List[float], rule: Dict[str, Any]) -> bool:
        """Évalue si la condition d'alerte est remplie."""
        if not metrics:
            return False
        
        threshold = rule["threshold"]
        operator = rule["operator"]
        
        # Utiliser la moyenne des métriques récentes
        avg_value = sum(metrics) / len(metrics)
        
        if operator == ">":
            return avg_value > threshold
        elif operator == "<":
            return avg_value < threshold
        elif operator == ">=":
            return avg_value >= threshold
        elif operator == "<=":
            return avg_value <= threshold
        elif operator == "==":
            return avg_value == threshold
        
        return False
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Vérifie si une règle est en cooldown."""
        if rule_name not in self.alert_cooldowns:
            return False
        
        last_triggered = self.alert_cooldowns[rule_name]
        cooldown_end = last_triggered + timedelta(minutes=15)  # Default cooldown
        
        return datetime.utcnow() < cooldown_end
    
    def _cleanup_cooldowns(self):
        """Nettoie les cooldowns expirés."""
        now = datetime.utcnow()
        expired_cooldowns = [
            rule_name for rule_name, last_triggered in self.alert_cooldowns.items()
            if now > last_triggered + timedelta(hours=1)  # Cleanup after 1 hour
        ]
        
        for rule_name in expired_cooldowns:
            del self.alert_cooldowns[rule_name]
    
    async def _trigger_alert(self, rule: Dict[str, Any], current_value: float):
        """Déclenche une alerte."""
        try:
            alert_id = f"{rule['name']}_{int(time.time())}"
            
            alert_data = {
                "alert_id": alert_id,
                "tenant_id": "system",  # À adapter selon la métrique
                "severity": rule["severity"],
                "title": f"Alerte: {rule['name']}",
                "description": f"La métrique {rule['metric']} a atteint {current_value:.2f}, "
                             f"seuil: {rule['threshold']} ({rule['operator']})",
                "current_value": current_value,
                "threshold_value": rule["threshold"],
                "triggered_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "rule": rule,
                    "metric_name": rule["metric"]
                }
            }
            
            # Sauvegarder l'alerte
            self.db.save_alert(alert_data)
            
            # Envoyer les notifications
            await self.notification_manager.send_alert(alert_data)
            
            # Ajouter au cooldown
            self.alert_cooldowns[rule["name"]] = datetime.utcnow()
            
            logger.warning(f"Alerte déclenchée: {alert_id}")
            
        except Exception as e:
            logger.error(f"Erreur déclenchement alerte: {e}")


# =============================================================================
# GÉNÉRATEUR DE RAPPORTS
# =============================================================================

class ReportGenerator:
    """Générateur de rapports automatisé."""
    
    def __init__(self, config: AutomationConfig, db: MonitoringDatabase):
        self.config = config
        self.db = db
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    async def generate_daily_report(self, tenant_id: str = None):
        """Génère un rapport quotidien."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=1)
            
            report_data = await self._collect_report_data(start_date, end_date, tenant_id)
            report_content = self._format_report(report_data, "daily")
            
            # Sauvegarder le rapport
            report_filename = f"daily_report_{end_date.strftime('%Y%m%d')}.html"
            if tenant_id:
                report_filename = f"daily_report_{tenant_id}_{end_date.strftime('%Y%m%d')}.html"
            
            report_path = self.reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Rapport quotidien généré: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Erreur génération rapport quotidien: {e}")
            return None
    
    async def generate_weekly_report(self, tenant_id: str = None):
        """Génère un rapport hebdomadaire."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(weeks=1)
            
            report_data = await self._collect_report_data(start_date, end_date, tenant_id)
            report_content = self._format_report(report_data, "weekly")
            
            report_filename = f"weekly_report_{end_date.strftime('%Y%m%d')}.html"
            if tenant_id:
                report_filename = f"weekly_report_{tenant_id}_{end_date.strftime('%Y%m%d')}.html"
            
            report_path = self.reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Rapport hebdomadaire généré: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Erreur génération rapport hebdomadaire: {e}")
            return None
    
    async def _collect_report_data(self, start_date: datetime, end_date: datetime, 
                                 tenant_id: str = None) -> Dict[str, Any]:
        """Collecte les données pour le rapport."""
        conn = sqlite3.connect(self.config.database_url.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        # Requête de base pour les métriques
        base_query = '''
            SELECT metric_name, AVG(metric_value), MIN(metric_value), MAX(metric_value), COUNT(*)
            FROM metrics 
            WHERE timestamp BETWEEN ? AND ?
        '''
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if tenant_id:
            base_query += ' AND tenant_id = ?'
            params.append(tenant_id)
        
        base_query += ' GROUP BY metric_name'
        
        cursor.execute(base_query, params)
        
        metrics_data = {}
        for row in cursor.fetchall():
            metrics_data[row[0]] = {
                'avg': round(row[1], 2),
                'min': round(row[2], 2),
                'max': round(row[3], 2),
                'count': row[4]
            }
        
        # Alertes pendant la période
        alert_query = '''
            SELECT severity, COUNT(*) 
            FROM alerts 
            WHERE triggered_at BETWEEN ? AND ?
        '''
        alert_params = [start_date.isoformat(), end_date.isoformat()]
        
        if tenant_id:
            alert_query += ' AND tenant_id = ?'
            alert_params.append(tenant_id)
        
        alert_query += ' GROUP BY severity'
        
        cursor.execute(alert_query, alert_params)
        
        alerts_data = {}
        for row in cursor.fetchall():
            alerts_data[row[0]] = row[1]
        
        conn.close()
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'tenant_id': tenant_id
            },
            'metrics': metrics_data,
            'alerts': alerts_data,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _format_report(self, data: Dict[str, Any], report_type: str) -> str:
        """Formate les données en rapport HTML."""
        tenant_info = f" - Tenant: {data['period']['tenant_id']}" if data['period']['tenant_id'] else ""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport {report_type.title()} - Monitoring Spotify AI Agent{tenant_info}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #1DB954; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .alert-critical {{ color: #ff0000; }}
        .alert-high {{ color: #ff8c00; }}
        .alert-medium {{ color: #ffd700; }}
        .alert-low {{ color: #90ee90; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Rapport {report_type.title()} - Monitoring Spotify AI Agent{tenant_info}</h1>
        <p>Période: {data['period']['start']} à {data['period']['end']}</p>
        <p>Généré le: {data['generated_at']}</p>
    </div>
    
    <div class="section">
        <h2>📊 Résumé des Métriques</h2>
        <table>
            <tr>
                <th>Métrique</th>
                <th>Moyenne</th>
                <th>Minimum</th>
                <th>Maximum</th>
                <th>Échantillons</th>
            </tr>
"""
        
        for metric_name, metric_data in data['metrics'].items():
            html_content += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{metric_data['avg']}</td>
                <td>{metric_data['min']}</td>
                <td>{metric_data['max']}</td>
                <td>{metric_data['count']}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>🚨 Résumé des Alertes</h2>
"""
        
        if data['alerts']:
            for severity, count in data['alerts'].items():
                html_content += f'<div class="metric alert-{severity}"><strong>{severity.upper()}:</strong> {count}</div>'
        else:
            html_content += '<p>Aucune alerte déclenchée pendant cette période.</p>'
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>📈 Points Clés</h2>
        <ul>
"""
        
        # Générer des insights automatiques
        if 'cpu_usage_percent' in data['metrics']:
            cpu_avg = data['metrics']['cpu_usage_percent']['avg']
            if cpu_avg > 80:
                html_content += f"<li>⚠️ Utilisation CPU élevée (moyenne: {cpu_avg}%)</li>"
            elif cpu_avg < 20:
                html_content += f"<li>✅ Utilisation CPU optimale (moyenne: {cpu_avg}%)</li>"
        
        if 'memory_usage_percent' in data['metrics']:
            memory_avg = data['metrics']['memory_usage_percent']['avg']
            if memory_avg > 85:
                html_content += f"<li>⚠️ Utilisation mémoire élevée (moyenne: {memory_avg}%)</li>"
        
        if 'error_rate' in data['metrics']:
            error_avg = data['metrics']['error_rate']['avg']
            if error_avg > 0.05:
                html_content += f"<li>🔴 Taux d'erreur préoccupant (moyenne: {error_avg:.3f})</li>"
            else:
                html_content += f"<li>✅ Taux d'erreur acceptable (moyenne: {error_avg:.3f})</li>"
        
        html_content += """
        </ul>
    </div>
    
    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>Rapport généré automatiquement par le système de monitoring Spotify AI Agent</p>
    </footer>
</body>
</html>
"""
        
        return html_content


# =============================================================================
# ORCHESTRATEUR PRINCIPAL
# =============================================================================

class MonitoringOrchestrator:
    """Orchestrateur principal du système de monitoring."""
    
    def __init__(self, config_path: str = None):
        self.config = AutomationConfig.from_file(config_path) if config_path else AutomationConfig()
        self.db = MonitoringDatabase(self.config.database_url)
        self.notification_manager = NotificationManager(self.config)
        self.collector = AutomatedCollector(self.config, self.db)
        self.alert_manager = AlertManager(self.config, self.db, self.notification_manager)
        self.report_generator = ReportGenerator(self.config, self.db)
        self.is_running = False
        
        # Gestionnaire de signaux pour arrêt gracieux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signal pour arrêt gracieux."""
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        self.stop()
    
    async def start(self):
        """Démarre l'orchestrateur."""
        self.is_running = True
        logger.info("Démarrage de l'orchestrateur de monitoring")
        
        try:
            # Planifier les tâches récurrentes
            self._schedule_tasks()
            
            # Démarrer la collecte en arrière-plan
            collection_task = asyncio.create_task(self.collector.start_collection())
            
            # Boucle principale de vérification des alertes
            while self.is_running:
                await self.alert_manager.check_alerts()
                schedule.run_pending()
                await asyncio.sleep(30)  # Vérifier les alertes toutes les 30 secondes
            
            # Arrêter la collecte
            self.collector.stop_collection()
            await collection_task
            
        except Exception as e:
            logger.error(f"Erreur dans l'orchestrateur: {e}")
            traceback.print_exc()
        finally:
            logger.info("Orchestrateur arrêté")
    
    def stop(self):
        """Arrête l'orchestrateur."""
        self.is_running = False
        self.collector.stop_collection()
    
    def _schedule_tasks(self):
        """Planifie les tâches récurrentes."""
        # Rapports quotidiens à 6h du matin
        schedule.every().day.at("06:00").do(self._generate_daily_reports)
        
        # Rapports hebdomadaires le lundi à 7h
        schedule.every().monday.at("07:00").do(self._generate_weekly_reports)
        
        # Nettoyage des anciennes données tous les jours à 2h
        schedule.every().day.at("02:00").do(self._cleanup_old_data)
        
        logger.info("Tâches récurrentes planifiées")
    
    def _generate_daily_reports(self):
        """Génère les rapports quotidiens."""
        async def generate():
            try:
                await self.report_generator.generate_daily_report()
                logger.info("Rapports quotidiens générés")
            except Exception as e:
                logger.error(f"Erreur génération rapports quotidiens: {e}")
        
        asyncio.create_task(generate())
    
    def _generate_weekly_reports(self):
        """Génère les rapports hebdomadaires."""
        async def generate():
            try:
                await self.report_generator.generate_weekly_report()
                logger.info("Rapports hebdomadaires générés")
            except Exception as e:
                logger.error(f"Erreur génération rapports hebdomadaires: {e}")
        
        asyncio.create_task(generate())
    
    def _cleanup_old_data(self):
        """Nettoie les anciennes données."""
        try:
            retention_date = datetime.utcnow() - timedelta(days=self.config.retention_days or 90)
            
            conn = sqlite3.connect(self.config.database_url.replace('sqlite:///', ''))
            cursor = conn.cursor()
            
            # Supprimer les anciennes métriques
            cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (retention_date.isoformat(),))
            metrics_deleted = cursor.rowcount
            
            # Supprimer les anciennes alertes résolues
            cursor.execute(
                'DELETE FROM alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?',
                (retention_date.isoformat(),)
            )
            alerts_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Nettoyage terminé: {metrics_deleted} métriques et {alerts_deleted} alertes supprimées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage données: {e}")


# =============================================================================
# INTERFACE LIGNE DE COMMANDE
# =============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description='Système d\'automatisation monitoring Spotify AI Agent')
    parser.add_argument('--config', '-c', help='Fichier de configuration')
    parser.add_argument('--debug', '-d', action='store_true', help='Mode debug')
    parser.add_argument('--command', choices=['start', 'collect', 'report', 'alert-test'], 
                       default='start', help='Commande à exécuter')
    parser.add_argument('--tenant-id', help='ID du tenant pour les rapports')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == 'start':
        # Démarrer l'orchestrateur complet
        orchestrator = MonitoringOrchestrator(args.config)
        asyncio.run(orchestrator.start())
    
    elif args.command == 'collect':
        # Lancer une collecte unique
        config = AutomationConfig.from_file(args.config) if args.config else AutomationConfig()
        db = MonitoringDatabase(config.database_url)
        collector = AutomatedCollector(config, db)
        
        async def run_collection():
            await collector._run_collection_cycle()
        
        asyncio.run(run_collection())
        logger.info("Collecte unique terminée")
    
    elif args.command == 'report':
        # Générer un rapport
        config = AutomationConfig.from_file(args.config) if args.config else AutomationConfig()
        db = MonitoringDatabase(config.database_url)
        generator = ReportGenerator(config, db)
        
        async def generate_report():
            report_path = await generator.generate_daily_report(args.tenant_id)
            if report_path:
                print(f"Rapport généré: {report_path}")
        
        asyncio.run(generate_report())
    
    elif args.command == 'alert-test':
        # Test des alertes
        config = AutomationConfig.from_file(args.config) if args.config else AutomationConfig()
        notification_manager = NotificationManager(config)
        
        test_alert = {
            "alert_id": "test_alert_001",
            "tenant_id": "test_tenant",
            "severity": "medium",
            "title": "Test d'alerte",
            "description": "Ceci est un test du système d'alerte",
            "triggered_at": datetime.utcnow().isoformat()
        }
        
        async def test_alert_notification():
            await notification_manager.send_alert(test_alert)
            print("Test d'alerte envoyé")
        
        asyncio.run(test_alert_notification())


if __name__ == "__main__":
    main()
