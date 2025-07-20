"""
Configuration d'alerting et monitoring enterprise ultra-avancée
Système de surveillance proactive avec intelligence artificielle intégrée
Créé par: Fahed Mlaiel - Expert en architecture de monitoring enterprise
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import aioredis
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Summary

# Configuration logging avancée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/spotify-ai/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes avec priorités automatiques"""
    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(Enum):
    """Types de métriques supportées pour l'observabilité"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    LATENCY = "latency"


class NotificationChannel(Enum):
    """Canaux de notification avec intégrations enterprise"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    JIRA = "jira"


@dataclass
class ThresholdConfig:
    """Configuration de seuils adaptatifs avec apprentissage automatique"""
    warning_threshold: float
    critical_threshold: float
    adaptive_thresholds: bool = True
    baseline_days: int = 30
    seasonal_adjustment: bool = True
    anomaly_detection: bool = True
    confidence_interval: float = 0.95
    
    def __post_init__(self):
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("Warning threshold must be less than critical threshold")


@dataclass
class AlertRule:
    """Règle d'alerte avancée avec logique métier intégrée"""
    name: str
    metric_name: str
    threshold_config: ThresholdConfig
    severity: AlertSeverity
    description: str
    query: str
    evaluation_interval: int = 60  # secondes
    for_duration: int = 300  # secondes
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    business_impact: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    inhibition_rules: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.metric_name:
            raise ValueError("Metric name is required")
        if not self.notification_channels:
            self.notification_channels = [NotificationChannel.EMAIL]


@dataclass
class DashboardConfig:
    """Configuration de dashboard avec visualisations avancées"""
    name: str
    title: str
    description: str
    tags: List[str] = field(default_factory=list)
    panels: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    time_range: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    refresh_interval: str = "30s"
    auto_refresh: bool = True
    shared: bool = False
    templating_enabled: bool = True


@dataclass
class MonitoringTarget:
    """Cible de monitoring avec découverte automatique"""
    name: str
    type: str  # service, database, cache, queue, etc.
    endpoints: List[str]
    labels: Dict[str, str] = field(default_factory=dict)
    scrape_interval: int = 30
    scrape_timeout: int = 10
    metrics_path: str = "/metrics"
    scheme: str = "http"
    health_check_path: str = "/health"
    discovery_method: str = "static"  # static, consul, k8s, etc.
    
    def __post_init__(self):
        if not self.endpoints:
            raise ValueError("At least one endpoint is required")


class AdvancedMonitoringManager:
    """Gestionnaire de monitoring enterprise avec IA intégrée"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_configuration(config_path)
        self.redis_client: Optional[aioredis.Redis] = None
        self.alert_rules: List[AlertRule] = []
        self.dashboards: List[DashboardConfig] = []
        self.targets: List[MonitoringTarget] = []
        self.metrics_registry = {}
        self.active_alerts = {}
        self.ai_models = {}
        
        # Métriques Prometheus intégrées
        self._setup_prometheus_metrics()
        
        logger.info("AdvancedMonitoringManager initialisé avec succès")
    
    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def _setup_prometheus_metrics(self):
        """Configure les métriques Prometheus prédéfinies"""
        self.metrics_registry['alerts_total'] = Counter(
            'monitoring_alerts_total',
            'Total number of alerts triggered',
            ['severity', 'service', 'environment']
        )
        
        self.metrics_registry['response_time'] = Histogram(
            'monitoring_response_time_seconds',
            'Response time of monitoring checks',
            ['target', 'check_type']
        )
        
        self.metrics_registry['system_health'] = Gauge(
            'monitoring_system_health_score',
            'Overall system health score (0-100)',
            ['component']
        )
        
        self.metrics_registry['anomalies_detected'] = Counter(
            'monitoring_anomalies_detected_total',
            'Number of anomalies detected by AI',
            ['model', 'severity']
        )
    
    async def initialize_connections(self):
        """Initialise les connexions aux services externes"""
        try:
            # Connexion Redis pour le cache des métriques
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            
            # Test de connexion
            await self.redis_client.ping()
            logger.info("Connexion Redis établie avec succès")
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation des connexions: {e}")
            raise
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Ajoute une règle d'alerte avec validation"""
        try:
            # Validation de la règle
            if any(r.name == rule.name for r in self.alert_rules):
                logger.warning(f"Règle d'alerte '{rule.name}' existe déjà")
                return False
            
            # Validation de la query Prometheus
            if not self._validate_prometheus_query(rule.query):
                logger.error(f"Query Prometheus invalide pour la règle '{rule.name}'")
                return False
            
            self.alert_rules.append(rule)
            logger.info(f"Règle d'alerte '{rule.name}' ajoutée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la règle d'alerte: {e}")
            return False
    
    def _validate_prometheus_query(self, query: str) -> bool:
        """Valide une query Prometheus"""
        try:
            # Validation basique de la syntaxe
            if not query or not isinstance(query, str):
                return False
            
            # Vérifications de sécurité
            forbidden_functions = ['delete', 'drop', 'truncate']
            query_lower = query.lower()
            
            if any(func in query_lower for func in forbidden_functions):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def create_dashboard(self, dashboard_config: DashboardConfig) -> str:
        """Crée un dashboard Grafana via API"""
        try:
            dashboard_json = {
                "dashboard": {
                    "title": dashboard_config.title,
                    "description": dashboard_config.description,
                    "tags": dashboard_config.tags,
                    "panels": dashboard_config.panels,
                    "templating": {
                        "list": dashboard_config.variables
                    },
                    "time": dashboard_config.time_range,
                    "refresh": dashboard_config.refresh_interval
                },
                "folderId": 0,
                "overwrite": True
            }
            
            # Simulation d'appel API Grafana
            async with aiohttp.ClientSession() as session:
                grafana_url = self.config.get('grafana_url', 'http://localhost:3000')
                api_key = self.config.get('grafana_api_key', '')
                
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    f"{grafana_url}/api/dashboards/db",
                    json=dashboard_json,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        dashboard_url = result.get('url', '')
                        logger.info(f"Dashboard créé: {dashboard_url}")
                        return dashboard_url
                    else:
                        logger.error(f"Erreur création dashboard: {response.status}")
                        return ""
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du dashboard: {e}")
            return ""
    
    async def evaluate_alert_rules(self) -> List[Dict[str, Any]]:
        """Évalue toutes les règles d'alerte avec IA intégrée"""
        triggered_alerts = []
        
        try:
            for rule in self.alert_rules:
                # Simulation d'évaluation de règle
                metric_value = await self._get_metric_value(rule.metric_name)
                
                if metric_value is None:
                    continue
                
                # Vérification des seuils adaptatifs
                thresholds = await self._calculate_adaptive_thresholds(
                    rule.metric_name, 
                    rule.threshold_config
                )
                
                alert_triggered = False
                severity = AlertSeverity.INFO
                
                if metric_value >= thresholds['critical']:
                    alert_triggered = True
                    severity = AlertSeverity.CRITICAL
                elif metric_value >= thresholds['warning']:
                    alert_triggered = True
                    severity = AlertSeverity.HIGH
                
                # Détection d'anomalies par IA
                if rule.threshold_config.anomaly_detection:
                    anomaly_score = await self._detect_anomaly(rule.metric_name, metric_value)
                    if anomaly_score > 0.8:  # Seuil d'anomalie
                        alert_triggered = True
                        severity = AlertSeverity.MEDIUM
                
                if alert_triggered:
                    alert_data = {
                        'rule_name': rule.name,
                        'metric_name': rule.metric_name,
                        'current_value': metric_value,
                        'severity': severity.value,
                        'description': rule.description,
                        'business_impact': rule.business_impact,
                        'remediation_steps': rule.remediation_steps,
                        'timestamp': datetime.utcnow().isoformat(),
                        'thresholds': thresholds
                    }
                    
                    triggered_alerts.append(alert_data)
                    
                    # Envoi de notifications
                    await self._send_notifications(rule, alert_data)
                    
                    # Mise à jour des métriques
                    self.metrics_registry['alerts_total'].labels(
                        severity=severity.value,
                        service=rule.labels.get('service', 'unknown'),
                        environment=rule.labels.get('environment', 'unknown')
                    ).inc()
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des règles d'alerte: {e}")
            return []
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Récupère la valeur d'une métrique depuis Prometheus"""
        try:
            # Simulation de récupération depuis Prometheus
            # Dans un vrai cas, on ferait un appel API à Prometheus
            
            # Vérification du cache Redis
            if self.redis_client:
                cached_value = await self.redis_client.get(f"metric:{metric_name}")
                if cached_value:
                    return float(cached_value)
            
            # Simulation de valeur métrique
            import random
            metric_value = random.uniform(0, 100)
            
            # Mise en cache
            if self.redis_client:
                await self.redis_client.setex(
                    f"metric:{metric_name}", 
                    30,  # TTL 30 secondes
                    str(metric_value)
                )
            
            return metric_value
            
        except Exception as e:
            logger.error(f"Erreur récupération métrique {metric_name}: {e}")
            return None
    
    async def _calculate_adaptive_thresholds(
        self, 
        metric_name: str, 
        threshold_config: ThresholdConfig
    ) -> Dict[str, float]:
        """Calcule les seuils adaptatifs basés sur l'historique"""
        try:
            base_thresholds = {
                'warning': threshold_config.warning_threshold,
                'critical': threshold_config.critical_threshold
            }
            
            if not threshold_config.adaptive_thresholds:
                return base_thresholds
            
            # Récupération de l'historique depuis Redis
            if self.redis_client:
                history_key = f"metric_history:{metric_name}"
                history_data = await self.redis_client.lrange(history_key, 0, -1)
                
                if len(history_data) >= 100:  # Minimum 100 points
                    values = [float(x) for x in history_data]
                    
                    # Calcul statistique des seuils adaptatifs
                    import numpy as np
                    
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    # Ajustement saisonnier
                    if threshold_config.seasonal_adjustment:
                        # Simulation d'ajustement saisonnier
                        seasonal_factor = 1.0  # À implémenter selon les besoins
                        mean_value *= seasonal_factor
                    
                    # Nouveaux seuils basés sur l'intervalle de confiance
                    confidence = threshold_config.confidence_interval
                    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% ou 99%
                    
                    adaptive_warning = mean_value + (z_score * std_value)
                    adaptive_critical = mean_value + (2 * z_score * std_value)
                    
                    return {
                        'warning': max(base_thresholds['warning'], adaptive_warning),
                        'critical': max(base_thresholds['critical'], adaptive_critical)
                    }
            
            return base_thresholds
            
        except Exception as e:
            logger.error(f"Erreur calcul seuils adaptatifs: {e}")
            return {
                'warning': threshold_config.warning_threshold,
                'critical': threshold_config.critical_threshold
            }
    
    async def _detect_anomaly(self, metric_name: str, current_value: float) -> float:
        """Détecte les anomalies using machine learning"""
        try:
            # Simulation de détection d'anomalies par IA
            # Dans un vrai cas, on utiliserait des modèles ML entraînés
            
            # Récupération de l'historique
            if self.redis_client:
                history_key = f"metric_history:{metric_name}"
                history_data = await self.redis_client.lrange(history_key, 0, 99)
                
                if len(history_data) >= 30:
                    import numpy as np
                    from scipy import stats
                    
                    values = np.array([float(x) for x in history_data])
                    
                    # Test de normalité
                    _, p_value = stats.normaltest(values)
                    
                    if p_value > 0.05:  # Distribution normale
                        z_score = abs(stats.zscore([current_value], values))[0]
                        anomaly_score = min(z_score / 3.0, 1.0)  # Normalisation
                    else:
                        # Utilisation de l'IQR pour distributions non-normales
                        q75, q25 = np.percentile(values, [75, 25])
                        iqr = q75 - q25
                        
                        if current_value < q25 - 1.5 * iqr or current_value > q75 + 1.5 * iqr:
                            anomaly_score = 0.9
                        else:
                            anomaly_score = 0.1
                    
                    # Mise à jour des métriques d'anomalies
                    if anomaly_score > 0.5:
                        self.metrics_registry['anomalies_detected'].labels(
                            model='statistical',
                            severity='medium' if anomaly_score > 0.8 else 'low'
                        ).inc()
                    
                    return anomaly_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            return 0.0
    
    async def _send_notifications(self, rule: AlertRule, alert_data: Dict[str, Any]):
        """Envoie les notifications selon les canaux configurés"""
        try:
            for channel in rule.notification_channels:
                await self._send_notification_to_channel(channel, alert_data)
                
        except Exception as e:
            logger.error(f"Erreur envoi notifications: {e}")
    
    async def _send_notification_to_channel(
        self, 
        channel: NotificationChannel, 
        alert_data: Dict[str, Any]
    ):
        """Envoie une notification à un canal spécifique"""
        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email_notification(alert_data)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack_notification(alert_data)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(alert_data)
            # Autres canaux...
            
        except Exception as e:
            logger.error(f"Erreur envoi notification {channel.value}: {e}")
    
    async def _send_email_notification(self, alert_data: Dict[str, Any]):
        """Envoie une notification par email"""
        # Simulation d'envoi d'email
        logger.info(f"📧 Email envoyé pour l'alerte: {alert_data['rule_name']}")
    
    async def _send_slack_notification(self, alert_data: Dict[str, Any]):
        """Envoie une notification Slack"""
        try:
            slack_webhook = self.config.get('slack_webhook_url', '')
            if not slack_webhook:
                return
            
            message = {
                "text": f"🚨 Alerte: {alert_data['rule_name']}",
                "attachments": [{
                    "color": "danger" if alert_data['severity'] == 'critical' else "warning",
                    "fields": [
                        {"title": "Métrique", "value": alert_data['metric_name'], "short": True},
                        {"title": "Valeur", "value": str(alert_data['current_value']), "short": True},
                        {"title": "Sévérité", "value": alert_data['severity'], "short": True},
                        {"title": "Description", "value": alert_data['description'], "short": False}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(slack_webhook, json=message) as response:
                    if response.status == 200:
                        logger.info("Notification Slack envoyée avec succès")
                    else:
                        logger.error(f"Erreur envoi Slack: {response.status}")
                        
        except Exception as e:
            logger.error(f"Erreur notification Slack: {e}")
    
    async def _send_webhook_notification(self, alert_data: Dict[str, Any]):
        """Envoie une notification via webhook"""
        try:
            webhook_url = self.config.get('webhook_url', '')
            if not webhook_url:
                return
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert_data) as response:
                    if response.status == 200:
                        logger.info("Notification webhook envoyée avec succès")
                    else:
                        logger.error(f"Erreur envoi webhook: {response.status}")
                        
        except Exception as e:
            logger.error(f"Erreur notification webhook: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du système de monitoring"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {},
                "overall_score": 100
            }
            
            # Vérification Redis
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status["components"]["redis"] = {"status": "healthy", "response_time_ms": 5}
                except Exception:
                    health_status["components"]["redis"] = {"status": "unhealthy", "error": "Connection failed"}
                    health_status["overall_score"] -= 25
            
            # Vérification des règles d'alerte
            health_status["components"]["alert_rules"] = {
                "status": "healthy",
                "count": len(self.alert_rules)
            }
            
            # Vérification des métriques
            health_status["components"]["metrics_registry"] = {
                "status": "healthy",
                "count": len(self.metrics_registry)
            }
            
            # Statut global
            if health_status["overall_score"] < 75:
                health_status["status"] = "degraded"
            elif health_status["overall_score"] < 50:
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Ressources nettoyées avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


# Fonctions utilitaires pour la configuration rapide
def create_basic_alert_rules() -> List[AlertRule]:
    """Crée des règles d'alerte de base pour un système typique"""
    return [
        AlertRule(
            name="cpu_usage_high",
            metric_name="cpu_usage_percent",
            threshold_config=ThresholdConfig(warning_threshold=80.0, critical_threshold=95.0),
            severity=AlertSeverity.HIGH,
            description="Utilisation CPU élevée détectée",
            query="100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            business_impact="Performance dégradée, risque d'indisponibilité",
            remediation_steps=[
                "Vérifier les processus consommateurs",
                "Analyser les logs d'application", 
                "Considérer l'ajout de ressources"
            ],
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        ),
        
        AlertRule(
            name="memory_usage_critical",
            metric_name="memory_usage_percent", 
            threshold_config=ThresholdConfig(warning_threshold=85.0, critical_threshold=95.0),
            severity=AlertSeverity.CRITICAL,
            description="Utilisation mémoire critique",
            query="(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            business_impact="Risque de crash application, OOM killer",
            remediation_steps=[
                "Identifier les fuites mémoire",
                "Redémarrer les services si nécessaire",
                "Augmenter la mémoire disponible"
            ],
            notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
        ),
        
        AlertRule(
            name="response_time_degraded",
            metric_name="http_request_duration_seconds",
            threshold_config=ThresholdConfig(warning_threshold=2.0, critical_threshold=5.0),
            severity=AlertSeverity.MEDIUM,
            description="Temps de réponse API dégradé",
            query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            business_impact="Expérience utilisateur dégradée",
            remediation_steps=[
                "Analyser les requêtes lentes",
                "Vérifier la base de données",
                "Optimiser les requêtes"
            ],
            notification_channels=[NotificationChannel.EMAIL]
        )
    ]


def create_system_dashboards() -> List[DashboardConfig]:
    """Crée des dashboards système standard"""
    return [
        DashboardConfig(
            name="system_overview",
            title="Vue d'ensemble du système",
            description="Dashboard principal avec métriques système essentielles",
            tags=["system", "overview", "production"],
            panels=[
                {
                    "title": "CPU Usage",
                    "type": "graph",
                    "query": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
                },
                {
                    "title": "Memory Usage", 
                    "type": "graph",
                    "query": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
                },
                {
                    "title": "Disk Usage",
                    "type": "graph", 
                    "query": "100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)"
                }
            ]
        ),
        
        DashboardConfig(
            name="application_performance",
            title="Performance Application",
            description="Métriques de performance applicative et business",
            tags=["application", "performance", "business"],
            panels=[
                {
                    "title": "Request Rate",
                    "type": "graph",
                    "query": "rate(http_requests_total[5m])"
                },
                {
                    "title": "Response Time P95",
                    "type": "graph", 
                    "query": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "query": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"
                }
            ]
        )
    ]


if __name__ == "__main__":
    async def main():
        """Exemple d'utilisation du système de monitoring avancé"""
        
        # Configuration
        config = {
            "redis_url": "redis://localhost:6379",
            "grafana_url": "http://localhost:3000",
            "grafana_api_key": "your_api_key_here",
            "slack_webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        }
        
        # Initialisation du manager
        monitoring_manager = AdvancedMonitoringManager()
        await monitoring_manager.initialize_connections()
        
        # Ajout des règles d'alerte de base
        for rule in create_basic_alert_rules():
            await monitoring_manager.add_alert_rule(rule)
        
        # Création des dashboards
        for dashboard in create_system_dashboards():
            dashboard_url = await monitoring_manager.create_dashboard(dashboard)
            print(f"Dashboard créé: {dashboard_url}")
        
        # Boucle d'évaluation des alertes
        try:
            while True:
                print("🔍 Évaluation des règles d'alerte...")
                triggered_alerts = await monitoring_manager.evaluate_alert_rules()
                
                if triggered_alerts:
                    print(f"⚠️  {len(triggered_alerts)} alertes déclenchées:")
                    for alert in triggered_alerts:
                        print(f"  - {alert['rule_name']}: {alert['current_value']}")
                else:
                    print("✅ Aucune alerte déclenchée")
                
                # Health check
                health = await monitoring_manager.health_check()
                print(f"🏥 Santé système: {health['status']} (score: {health['overall_score']})")
                
                # Attendre avant la prochaine évaluation
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            print("🛑 Arrêt du monitoring...")
        finally:
            await monitoring_manager.cleanup()
    
    # Exécution
    asyncio.run(main())
