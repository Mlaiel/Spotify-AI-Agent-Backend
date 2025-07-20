#!/usr/bin/env python3
"""
Script de Monitoring Avancé en Temps Réel
=========================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, Architecte Microservices

Ce script implémente un système de monitoring en temps réel avec
détection d'anomalies avancée et alertes intelligentes.
"""

import asyncio
import logging
import json
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yaml
import signal
from pathlib import Path
import aioredis
import aiohttp
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Métriques Prometheus
METRICS = {
    'alerts_total': Counter('spotify_ai_monitoring_alerts_total', 'Nombre total d\'alertes générées', ['severity', 'type']),
    'detection_time': Histogram('spotify_ai_detection_time_seconds', 'Temps de détection en secondes', ['detector_type']),
    'system_health': Gauge('spotify_ai_system_health_score', 'Score de santé du système', ['component']),
    'active_detectors': Gauge('spotify_ai_active_detectors', 'Nombre de détecteurs actifs'),
    'processing_rate': Gauge('spotify_ai_processing_rate_per_second', 'Taux de traitement par seconde')
}

class MonitoringConfig:
    """Configuration du monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.validate_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier"""
        default_config = {
            'monitoring': {
                'interval_seconds': 30,
                'batch_size': 100,
                'enable_prometheus': True,
                'prometheus_port': 8000,
                'log_level': 'INFO'
            },
            'detectors': {
                'ml_anomaly': {
                    'enabled': True,
                    'sensitivity': 0.8,
                    'model_path': '/tmp/models/anomaly_detector.pkl'
                },
                'threshold': {
                    'enabled': True,
                    'cpu_threshold': 85.0,
                    'memory_threshold': 90.0,
                    'disk_threshold': 95.0
                },
                'security': {
                    'enabled': True,
                    'max_failed_logins': 5,
                    'suspicious_ip_threshold': 100
                }
            },
            'notifications': {
                'slack': {
                    'enabled': True,
                    'webhook_url': None,
                    'channel': '#alerts'
                },
                'email': {
                    'enabled': False,
                    'smtp_server': 'localhost',
                    'smtp_port': 587,
                    'recipients': []
                }
            },
            'data_sources': {
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                },
                'prometheus': {
                    'url': 'http://localhost:9090'
                },
                'elasticsearch': {
                    'hosts': ['localhost:9200'],
                    'index_pattern': 'logs-*'
                }
            },
            'storage': {
                'alerts_retention_days': 30,
                'metrics_retention_days': 7,
                'export_formats': ['json', 'csv']
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Fusion profonde des configurations
                default_config = self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.error(f"Erreur chargement config {config_path}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Fusion profonde de deux dictionnaires"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def validate_config(self):
        """Valide la configuration"""
        required_sections = ['monitoring', 'detectors', 'data_sources']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Section de configuration manquante: {section}")
        
        # Validation des intervalles
        interval = self.config['monitoring'].get('interval_seconds', 30)
        if interval < 10 or interval > 3600:
            raise ValueError("L'intervalle de monitoring doit être entre 10 et 3600 secondes")
        
        logger.info("Configuration validée avec succès")

class AlertManager:
    """Gestionnaire d'alertes avancé"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_cache = {}
        self.notification_history = []
        self.alert_correlations = {}
        
    async def process_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Traite une nouvelle alerte"""
        try:
            # Déduplication
            alert_key = self._generate_alert_key(alert_data)
            if await self._is_duplicate_alert(alert_key, alert_data):
                logger.debug(f"Alerte dupliquée ignorée: {alert_key}")
                return False
            
            # Enrichissement de l'alerte
            enriched_alert = await self._enrich_alert(alert_data)
            
            # Corrélation avec d'autres alertes
            correlations = await self._correlate_alerts(enriched_alert)
            enriched_alert['correlations'] = correlations
            
            # Stockage
            await self._store_alert(enriched_alert)
            
            # Notification
            await self._send_notifications(enriched_alert)
            
            # Métriques
            METRICS['alerts_total'].labels(
                severity=enriched_alert.get('severity', 'unknown'),
                type=enriched_alert.get('type', 'unknown')
            ).inc()
            
            logger.info(f"Alerte traitée: {alert_key}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur traitement alerte: {e}")
            return False
    
    def _generate_alert_key(self, alert_data: Dict[str, Any]) -> str:
        """Génère une clé unique pour l'alerte"""
        source = alert_data.get('source', {})
        system = source.get('system', 'unknown')
        component = source.get('component', 'unknown')
        message_hash = hash(alert_data.get('message', ''))
        
        return f"{system}:{component}:{message_hash}"
    
    async def _is_duplicate_alert(self, alert_key: str, alert_data: Dict[str, Any]) -> bool:
        """Vérifie si l'alerte est un doublon"""
        if alert_key in self.alert_cache:
            cached_alert = self.alert_cache[alert_key]
            time_diff = datetime.now() - cached_alert['timestamp']
            
            # Considérer comme doublon si moins de 5 minutes
            if time_diff < timedelta(minutes=5):
                return True
        
        # Mettre en cache la nouvelle alerte
        self.alert_cache[alert_key] = {
            'timestamp': datetime.now(),
            'data': alert_data
        }
        
        return False
    
    async def _enrich_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit l'alerte avec des informations contextuelles"""
        enriched = alert_data.copy()
        
        # Ajouter des métadonnées système
        enriched['processing_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'processor_version': '2.1.0',
            'enrichment_level': 'advanced'
        }
        
        # Enrichissement basé sur le type d'alerte
        alert_type = alert_data.get('type', 'unknown')
        
        if alert_type == 'performance':
            enriched = await self._enrich_performance_alert(enriched)
        elif alert_type == 'security':
            enriched = await self._enrich_security_alert(enriched)
        elif alert_type == 'anomaly':
            enriched = await self._enrich_anomaly_alert(enriched)
        
        # Calcul du score d'impact
        enriched['impact_score'] = self._calculate_impact_score(enriched)
        
        # Ajout de recommandations automatiques
        enriched['auto_recommendations'] = self._generate_auto_recommendations(enriched)
        
        return enriched
    
    async def _enrich_performance_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit une alerte de performance"""
        # Récupérer des métriques supplémentaires
        alert['performance_context'] = {
            'system_load': await self._get_current_system_load(),
            'active_connections': await self._get_active_connections(),
            'resource_utilization': await self._get_resource_utilization()
        }
        
        return alert
    
    async def _enrich_security_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit une alerte de sécurité"""
        # Analyse de la source IP si disponible
        context = alert.get('context', {})
        if 'ip_address' in context:
            ip_info = await self._analyze_ip_address(context['ip_address'])
            alert['security_context'] = ip_info
        
        return alert
    
    async def _enrich_anomaly_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit une alerte d'anomalie"""
        # Ajouter le contexte historique
        alert['anomaly_context'] = {
            'historical_frequency': await self._get_anomaly_frequency(),
            'similar_patterns': await self._find_similar_anomalies(alert),
            'baseline_deviation': await self._calculate_baseline_deviation(alert)
        }
        
        return alert
    
    def _calculate_impact_score(self, alert: Dict[str, Any]) -> float:
        """Calcule un score d'impact pour l'alerte"""
        score = 0.0
        
        # Score basé sur la sévérité
        severity_scores = {
            'critical': 1.0,
            'high': 0.8, 
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }
        score += severity_scores.get(alert.get('severity', 'low'), 0.2)
        
        # Score basé sur la confiance
        confidence = alert.get('confidence_score', 0.5)
        score *= confidence
        
        # Score basé sur l'impact utilisateur
        user_impact = alert.get('context', {}).get('user_impact', {})
        impact_level = user_impact.get('impact_level', 'none')
        impact_scores = {
            'severe': 1.0,
            'significant': 0.8,
            'moderate': 0.6,
            'minimal': 0.4,
            'none': 0.2
        }
        score *= impact_scores.get(impact_level, 0.2)
        
        return min(score, 1.0)
    
    def _generate_auto_recommendations(self, alert: Dict[str, Any]) -> List[str]:
        """Génère des recommandations automatiques"""
        recommendations = []
        
        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'low')
        
        if alert_type == 'performance' and severity in ['critical', 'high']:
            recommendations.extend([
                "Vérifier la charge système actuelle",
                "Analyser les processus consommant le plus de ressources",
                "Considérer l'ajout de ressources ou la répartition de charge"
            ])
        
        elif alert_type == 'security':
            recommendations.extend([
                "Bloquer immédiatement les IPs suspectes",
                "Auditer les logs de sécurité",
                "Notifier l'équipe de sécurité"
            ])
        
        elif alert_type == 'anomaly':
            recommendations.extend([
                "Analyser les patterns de données récents",
                "Vérifier les changements de configuration",
                "Comparer avec les anomalies historiques"
            ])
        
        # Recommandations générales basées sur la sévérité
        if severity == 'critical':
            recommendations.insert(0, "Action immédiate requise - Escalader vers l'équipe d'astreinte")
        
        return recommendations
    
    async def _correlate_alerts(self, alert: Dict[str, Any]) -> List[str]:
        """Corrèle l'alerte avec d'autres alertes récentes"""
        correlations = []
        
        # Rechercher des alertes similaires dans les dernières heures
        current_time = datetime.now()
        time_window = timedelta(hours=2)
        
        for cached_key, cached_data in self.alert_cache.items():
            if current_time - cached_data['timestamp'] <= time_window:
                if self._are_alerts_correlated(alert, cached_data['data']):
                    correlations.append(cached_key)
        
        return correlations
    
    def _are_alerts_correlated(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> bool:
        """Détermine si deux alertes sont corrélées"""
        # Même système source
        if alert1.get('source', {}).get('system') == alert2.get('source', {}).get('system'):
            return True
        
        # Même composant affecté
        services1 = set(alert1.get('context', {}).get('affected_services', []))
        services2 = set(alert2.get('context', {}).get('affected_services', []))
        if services1.intersection(services2):
            return True
        
        # Même tenant
        tenant1 = alert1.get('context', {}).get('tenant_id')
        tenant2 = alert2.get('context', {}).get('tenant_id')
        if tenant1 and tenant2 and tenant1 == tenant2:
            return True
        
        return False
    
    async def _store_alert(self, alert: Dict[str, Any]):
        """Stocke l'alerte de manière persistante"""
        try:
            # Stockage en fichier JSON (dans un vrai système, utiliser une DB)
            alerts_dir = Path("/tmp/alerts")
            alerts_dir.mkdir(exist_ok=True)
            
            alert_file = alerts_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            
            with open(alert_file, 'w') as f:
                json.dump(alert, f, indent=2, default=str)
            
            logger.debug(f"Alerte stockée: {alert_file}")
            
        except Exception as e:
            logger.error(f"Erreur stockage alerte: {e}")
    
    async def _send_notifications(self, alert: Dict[str, Any]):
        """Envoie les notifications pour l'alerte"""
        try:
            severity = alert.get('severity', 'low')
            
            # Notification Slack
            if self.config.config['notifications']['slack']['enabled']:
                await self._send_slack_notification(alert)
            
            # Notification email pour les alertes critiques
            if severity == 'critical' and self.config.config['notifications']['email']['enabled']:
                await self._send_email_notification(alert)
            
            # Webhook personnalisé
            await self._send_webhook_notification(alert)
            
        except Exception as e:
            logger.error(f"Erreur envoi notifications: {e}")
    
    async def _send_slack_notification(self, alert: Dict[str, Any]):
        """Envoie une notification Slack"""
        slack_config = self.config.config['notifications']['slack']
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("URL Slack webhook non configurée")
            return
        
        # Formatage du message Slack
        color_map = {
            'critical': '#ff0000',
            'high': '#ff8800',
            'medium': '#ffaa00',
            'low': '#00aa00',
            'info': '#0088aa'
        }
        
        severity = alert.get('severity', 'info')
        color = color_map.get(severity, '#808080')
        
        slack_message = {
            "channel": slack_config.get('channel', '#alerts'),
            "username": "Spotify AI Monitor",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color,
                    "title": f"Alerte {severity.upper()}",
                    "text": alert.get('message', 'Alerte sans message'),
                    "fields": [
                        {
                            "title": "Source",
                            "value": f"{alert.get('source', {}).get('system', 'Unknown')} - {alert.get('source', {}).get('component', 'Unknown')}",
                            "short": True
                        },
                        {
                            "title": "Confiance",
                            "value": f"{alert.get('confidence_score', 0) * 100:.1f}%",
                            "short": True
                        },
                        {
                            "title": "Impact",
                            "value": f"Score: {alert.get('impact_score', 0):.2f}",
                            "short": True
                        },
                        {
                            "title": "Recommandations",
                            "value": "\n".join(alert.get('auto_recommendations', [])[:3]),
                            "short": False
                        }
                    ],
                    "timestamp": int(datetime.now().timestamp())
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status == 200:
                        logger.debug("Notification Slack envoyée avec succès")
                    else:
                        logger.error(f"Erreur Slack: {response.status}")
        
        except Exception as e:
            logger.error(f"Erreur envoi Slack: {e}")
    
    async def _send_webhook_notification(self, alert: Dict[str, Any]):
        """Envoie une notification webhook générique"""
        # Implémentation webhook personnalisé
        webhook_payload = {
            "alert": alert,
            "timestamp": datetime.now().isoformat(),
            "source": "spotify-ai-monitoring"
        }
        
        # Dans un vrai système, envoyer vers les endpoints configurés
        logger.debug(f"Webhook payload préparé: {len(json.dumps(webhook_payload))} bytes")
    
    # Méthodes auxiliaires pour l'enrichissement
    async def _get_current_system_load(self) -> Dict[str, float]:
        """Récupère la charge système actuelle"""
        return {
            "cpu_percent": 45.2,
            "memory_percent": 67.8,
            "disk_io_wait": 2.1
        }
    
    async def _get_active_connections(self) -> int:
        """Récupère le nombre de connexions actives"""
        return 1547
    
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Récupère l'utilisation des ressources"""
        return {
            "cpu_cores_used": 6.7,
            "memory_gb_used": 12.3,
            "network_mbps": 156.4
        }
    
    async def _analyze_ip_address(self, ip_address: str) -> Dict[str, Any]:
        """Analyse une adresse IP"""
        return {
            "is_internal": ip_address.startswith('192.168.') or ip_address.startswith('10.'),
            "reputation_score": 0.8,
            "geographic_location": "Unknown",
            "known_threats": []
        }
    
    async def _get_anomaly_frequency(self) -> float:
        """Récupère la fréquence des anomalies"""
        return 0.05  # 5% des mesures sont des anomalies
    
    async def _find_similar_anomalies(self, alert: Dict[str, Any]) -> List[str]:
        """Trouve des anomalies similaires"""
        return ["anomaly_2024_001", "anomaly_2024_045"]
    
    async def _calculate_baseline_deviation(self, alert: Dict[str, Any]) -> float:
        """Calcule la déviation par rapport à la baseline"""
        return 2.34  # Déviation en nombre d'écarts-types

class MonitoringOrchestrator:
    """Orchestrateur principal du monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config = MonitoringConfig(config_path)
        self.alert_manager = AlertManager(self.config)
        self.is_running = False
        self.statistics = {
            'start_time': None,
            'alerts_processed': 0,
            'errors_count': 0,
            'avg_processing_time': 0.0
        }
        
        # Gestionnaire de signaux pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre"""
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        self.is_running = False
    
    async def start(self):
        """Démarre le monitoring"""
        logger.info("Démarrage du monitoring Spotify AI Agent...")
        
        try:
            # Démarrage du serveur Prometheus
            if self.config.config['monitoring']['enable_prometheus']:
                prometheus_port = self.config.config['monitoring']['prometheus_port']
                start_http_server(prometheus_port)
                logger.info(f"Serveur Prometheus démarré sur le port {prometheus_port}")
            
            # Initialisation des statistiques
            self.statistics['start_time'] = datetime.now()
            self.is_running = True
            
            # Mise à jour des métriques initiales
            METRICS['active_detectors'].set(len(self.config.config['detectors']))
            
            # Boucle principale de monitoring
            await self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        interval = self.config.config['monitoring']['interval_seconds']
        
        while self.is_running:
            loop_start = datetime.now()
            
            try:
                # Collecte et analyse des données
                await self._collect_and_analyze()
                
                # Mise à jour des métriques de performance
                processing_time = (datetime.now() - loop_start).total_seconds()
                self._update_statistics(processing_time)
                
                # Métriques Prometheus
                METRICS['processing_rate'].set(1.0 / processing_time if processing_time > 0 else 0)
                
                # Attendre jusqu'au prochain cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.statistics['errors_count'] += 1
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                await asyncio.sleep(5)  # Courte pause avant retry
    
    async def _collect_and_analyze(self):
        """Collecte et analyse les données"""
        # Simulation de collecte de données
        # Dans un vrai système, ici on interrogerait les sources de données réelles
        
        sample_alerts = await self._generate_sample_alerts()
        
        for alert_data in sample_alerts:
            start_time = datetime.now()
            
            success = await self.alert_manager.process_alert(alert_data)
            
            if success:
                self.statistics['alerts_processed'] += 1
            
            # Métriques de temps de détection
            detection_time = (datetime.now() - start_time).total_seconds()
            detector_type = alert_data.get('source', {}).get('detector_type', 'unknown')
            METRICS['detection_time'].labels(detector_type=detector_type).observe(detection_time)
    
    async def _generate_sample_alerts(self) -> List[Dict[str, Any]]:
        """Génère des alertes d'exemple pour la démonstration"""
        import random
        
        # Retourner parfois aucune alerte
        if random.random() < 0.7:  # 70% de chance de ne pas avoir d'alerte
            return []
        
        # Types d'alertes possibles
        alert_types = [
            {
                "alert_id": f"cpu-alert-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": random.choice(['medium', 'high']),
                "source": {
                    "system": "threshold_detector",
                    "component": "cpu_monitor",
                    "detector_type": "threshold"
                },
                "type": "performance",
                "message": f"CPU usage at {random.randint(80, 95)}% for 5 minutes",
                "confidence_score": random.uniform(0.7, 0.95),
                "context": {
                    "tenant_id": "spotify-tenant-001",
                    "environment": "production",
                    "affected_services": ["audio-processing"]
                }
            },
            {
                "alert_id": f"anomaly-alert-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": random.choice(['low', 'medium']),
                "source": {
                    "system": "ml_detector",
                    "component": "behavior_analyzer",
                    "detector_type": "ml_based"
                },
                "type": "anomaly",
                "message": "Unusual user behavior pattern detected",
                "confidence_score": random.uniform(0.5, 0.8),
                "context": {
                    "tenant_id": "spotify-tenant-002",
                    "environment": "production"
                }
            }
        ]
        
        # Retourner 1-2 alertes aléatoirement
        num_alerts = random.randint(1, 2)
        return random.sample(alert_types, min(num_alerts, len(alert_types)))
    
    def _update_statistics(self, processing_time: float):
        """Met à jour les statistiques de performance"""
        total_alerts = self.statistics['alerts_processed']
        
        if total_alerts > 0:
            current_avg = self.statistics['avg_processing_time']
            new_avg = (current_avg * (total_alerts - 1) + processing_time) / total_alerts
            self.statistics['avg_processing_time'] = new_avg
        
        # Mise à jour des métriques de santé système
        uptime_hours = (datetime.now() - self.statistics['start_time']).total_seconds() / 3600
        error_rate = self.statistics['errors_count'] / max(total_alerts, 1)
        
        health_score = max(0.0, 1.0 - error_rate - (0.1 if uptime_hours > 24 else 0))
        METRICS['system_health'].labels(component='monitoring').set(health_score)
    
    async def _cleanup(self):
        """Nettoyage des ressources"""
        logger.info("Nettoyage des ressources en cours...")
        
        try:
            # Affichage des statistiques finales
            uptime = datetime.now() - self.statistics['start_time']
            logger.info(f"Statistiques finales:")
            logger.info(f"  - Durée de fonctionnement: {uptime}")
            logger.info(f"  - Alertes traitées: {self.statistics['alerts_processed']}")
            logger.info(f"  - Erreurs: {self.statistics['errors_count']}")
            logger.info(f"  - Temps de traitement moyen: {self.statistics['avg_processing_time']:.3f}s")
            
            # Sauvegarder les statistiques
            stats_file = Path("/tmp/monitoring_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            
            logger.info("Nettoyage terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du monitoring"""
        uptime = datetime.now() - self.statistics['start_time'] if self.statistics['start_time'] else timedelta(0)
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime.total_seconds(),
            'configuration': {
                'interval_seconds': self.config.config['monitoring']['interval_seconds'],
                'detectors_enabled': len([d for d in self.config.config['detectors'].values() if d.get('enabled', False)]),
                'prometheus_enabled': self.config.config['monitoring']['enable_prometheus']
            },
            'statistics': self.statistics,
            'health': {
                'error_rate': self.statistics['errors_count'] / max(self.statistics['alerts_processed'], 1),
                'avg_processing_time': self.statistics['avg_processing_time']
            }
        }

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Monitoring Avancé Spotify AI Agent')
    parser.add_argument('--config', '-c', help='Chemin vers le fichier de configuration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    parser.add_argument('--status', action='store_true', help='Afficher le statut et quitter')
    parser.add_argument('--test', action='store_true', help='Mode test avec alertes simulées')
    
    args = parser.parse_args()
    
    # Configuration du niveau de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Création de l'orchestrateur
        orchestrator = MonitoringOrchestrator(args.config)
        
        if args.status:
            # Afficher le statut (nécessite une instance en cours)
            status = await orchestrator.get_status()
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        if args.test:
            logger.info("Mode test activé - Génération d'alertes simulées")
        
        # Démarrage du monitoring
        await orchestrator.start()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        return 0
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
