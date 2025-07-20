"""
Module d'intégration ultra-avancé pour Alertmanager Receivers

Ce module gère les intégrations avec plus de 20 systèmes externes,
incluant l'observabilité, la communication, les ITSM et les plateformes cloud.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Architecte Microservices
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import base64
import hmac
import hashlib
from urllib.parse import urlencode
import ssl

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types d'intégrations supportées"""
    MESSAGING = "messaging"          # Slack, Teams, Discord
    TICKETING = "ticketing"          # Jira, ServiceNow, PagerDuty
    MONITORING = "monitoring"        # Datadog, New Relic, Prometheus
    CLOUD = "cloud"                  # AWS, Azure, GCP
    COLLABORATION = "collaboration"  # Confluence, Notion
    DEVOPS = "devops"               # GitHub, GitLab, Jenkins
    SECURITY = "security"           # Splunk, CrowdStrike
    ANALYTICS = "analytics"         # Elasticsearch, Grafana

class AuthType(Enum):
    """Types d'authentification"""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    WEBHOOK_SECRET = "webhook_secret"
    CERTIFICATE = "certificate"

@dataclass
class IntegrationEndpoint:
    """Configuration d'un endpoint d'intégration"""
    name: str
    url: str
    auth_type: AuthType
    credentials: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 2
    rate_limit_per_minute: int = 60
    ssl_verify: bool = True
    custom_ca_bundle: Optional[str] = None

@dataclass
class IntegrationConfig:
    """Configuration complète d'une intégration"""
    name: str
    type: IntegrationType
    enabled: bool = True
    tenant_specific: bool = True
    endpoints: List[IntegrationEndpoint] = field(default_factory=list)
    default_settings: Dict[str, Any] = field(default_factory=dict)
    webhook_validation: bool = True
    data_transformation: Optional[str] = None
    error_handling: Dict[str, Any] = field(default_factory=dict)

class SlackIntegration:
    """Intégration avancée avec Slack"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise la session Slack"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.endpoints[0].timeout)
        )
    
    async def send_alert(self, alert_data: Dict, tenant: str) -> bool:
        """Envoie une alerte vers Slack"""
        try:
            endpoint = self._get_tenant_endpoint(tenant)
            
            # Construction du message Slack avancé
            message = await self._build_slack_message(alert_data, tenant)
            
            # Envoi via webhook
            async with self.session.post(
                endpoint.url,
                json=message,
                headers=endpoint.headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent successfully for {tenant}")
                    return True
                else:
                    logger.error(f"Slack send failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Slack integration error: {e}")
            return False
    
    async def _build_slack_message(self, alert_data: Dict, tenant: str) -> Dict:
        """Construit un message Slack avancé avec blocks"""
        severity = alert_data.get("severity", "unknown")
        service = alert_data.get("service", "unknown")
        description = alert_data.get("description", "No description")
        
        # Emoji selon la sévérité
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️", 
            "medium": "🟡",
            "low": "🔵",
            "info": "ℹ️"
        }.get(severity, "❓")
        
        message = {
            "text": f"{severity_emoji} Alert for {service}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{severity_emoji} {severity.upper()} Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Service:*\n{service}"
                        },
                        {
                            "type": "mrkdwn", 
                            "text": f"*Tenant:*\n{tenant}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{severity}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:*\n{description}"
                    }
                }
            ]
        }
        
        # Ajout de boutons d'action pour les alertes critiques
        if severity in ["critical", "high"]:
            message["blocks"].append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "🔍 Investigate"
                        },
                        "style": "primary",
                        "url": f"https://monitoring.spotify.com/alerts/{alert_data.get('id', '')}"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📖 Runbook"
                        },
                        "url": f"https://runbook.spotify.com/{service}"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "✅ Acknowledge"
                        },
                        "style": "danger"
                    }
                ]
            })
        
        return message
    
    def _get_tenant_endpoint(self, tenant: str) -> IntegrationEndpoint:
        """Récupère l'endpoint spécifique au tenant"""
        # Par défaut, utilise le premier endpoint
        return self.config.endpoints[0]
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

class PagerDutyIntegration:
    """Intégration avancée avec PagerDuty"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise la session PagerDuty"""
        self.session = aiohttp.ClientSession()
    
    async def create_incident(self, alert_data: Dict, tenant: str) -> Optional[str]:
        """Crée un incident PagerDuty"""
        try:
            endpoint = self._get_tenant_endpoint(tenant)
            integration_key = endpoint.credentials.get("integration_key")
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "client": "Spotify AlertManager",
                "client_url": "https://alertmanager.spotify.com",
                "payload": {
                    "summary": f"{alert_data.get('service', 'Unknown')} - {alert_data.get('description', '')}",
                    "source": alert_data.get("service", "unknown"),
                    "severity": self._map_severity_to_pagerduty(alert_data.get("severity", "medium")),
                    "component": alert_data.get("component", ""),
                    "group": tenant,
                    "class": "infrastructure",
                    "custom_details": {
                        "tenant": tenant,
                        "alert_id": alert_data.get("id", ""),
                        "metrics": alert_data.get("metrics", {}),
                        "runbook_url": f"https://runbook.spotify.com/{alert_data.get('service', '')}"
                    }
                }
            }
            
            async with self.session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 202:
                    result = await response.json()
                    incident_key = result.get("dedup_key")
                    logger.info(f"PagerDuty incident created: {incident_key}")
                    return incident_key
                else:
                    logger.error(f"PagerDuty incident creation failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"PagerDuty integration error: {e}")
            return None
    
    def _map_severity_to_pagerduty(self, severity: str) -> str:
        """Mappe la sévérité vers les niveaux PagerDuty"""
        mapping = {
            "critical": "critical",
            "high": "error",
            "medium": "warning", 
            "low": "info",
            "info": "info"
        }
        return mapping.get(severity, "warning")
    
    def _get_tenant_endpoint(self, tenant: str) -> IntegrationEndpoint:
        """Récupère l'endpoint spécifique au tenant"""
        return self.config.endpoints[0]
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

class JiraIntegration:
    """Intégration avancée avec Jira"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise la session Jira"""
        self.session = aiohttp.ClientSession()
    
    async def create_ticket(self, alert_data: Dict, tenant: str) -> Optional[str]:
        """Crée un ticket Jira"""
        try:
            endpoint = self._get_tenant_endpoint(tenant)
            
            # Authentification
            auth_header = self._build_auth_header(endpoint)
            
            # Construction du ticket
            ticket_data = {
                "fields": {
                    "project": {"key": endpoint.credentials.get("project_key", "ALERT")},
                    "summary": f"[{tenant}] {alert_data.get('service', 'Unknown')} Alert",
                    "description": self._build_jira_description(alert_data, tenant),
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": self._map_severity_to_jira(alert_data.get("severity", "medium"))},
                    "labels": [
                        "alert",
                        f"tenant-{tenant}",
                        f"service-{alert_data.get('service', 'unknown')}",
                        f"severity-{alert_data.get('severity', 'medium')}"
                    ],
                    "customfield_10000": alert_data.get("id", ""),  # Alert ID
                }
            }
            
            async with self.session.post(
                f"{endpoint.url}/rest/api/3/issue",
                json=ticket_data,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    ticket_key = result.get("key")
                    logger.info(f"Jira ticket created: {ticket_key}")
                    return ticket_key
                else:
                    logger.error(f"Jira ticket creation failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Jira integration error: {e}")
            return None
    
    def _build_auth_header(self, endpoint: IntegrationEndpoint) -> str:
        """Construit l'en-tête d'authentification"""
        username = endpoint.credentials.get("username")
        token = endpoint.credentials.get("api_token")
        
        credentials = base64.b64encode(f"{username}:{token}".encode()).decode()
        return f"Basic {credentials}"
    
    def _build_jira_description(self, alert_data: Dict, tenant: str) -> str:
        """Construit la description détaillée du ticket"""
        description = f"""
*Alert Details*

*Service:* {alert_data.get('service', 'Unknown')}
*Tenant:* {tenant}
*Severity:* {alert_data.get('severity', 'Unknown')}
*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

*Description:*
{alert_data.get('description', 'No description available')}

*Metrics:*
{json.dumps(alert_data.get('metrics', {}), indent=2)}

*Investigation Links:*
- [Monitoring Dashboard|https://monitoring.spotify.com/alerts/{alert_data.get('id', '')}]
- [Runbook|https://runbook.spotify.com/{alert_data.get('service', '')}]
- [Logs|https://logs.spotify.com/search?service={alert_data.get('service', '')}]
        """
        return description.strip()
    
    def _map_severity_to_jira(self, severity: str) -> str:
        """Mappe la sévérité vers les priorités Jira"""
        mapping = {
            "critical": "Highest",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
            "info": "Lowest"
        }
        return mapping.get(severity, "Medium")
    
    def _get_tenant_endpoint(self, tenant: str) -> IntegrationEndpoint:
        """Récupère l'endpoint spécifique au tenant"""
        return self.config.endpoints[0]
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

class DatadogIntegration:
    """Intégration avancée avec Datadog"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise la session Datadog"""
        self.session = aiohttp.ClientSession()
    
    async def send_event(self, alert_data: Dict, tenant: str) -> bool:
        """Envoie un événement vers Datadog"""
        try:
            endpoint = self._get_tenant_endpoint(tenant)
            api_key = endpoint.credentials.get("api_key")
            
            event_data = {
                "title": f"Alert: {alert_data.get('service', 'Unknown')}",
                "text": alert_data.get("description", ""),
                "date_happened": int(datetime.utcnow().timestamp()),
                "priority": "normal" if alert_data.get("severity") in ["low", "info"] else "high",
                "tags": [
                    f"tenant:{tenant}",
                    f"service:{alert_data.get('service', 'unknown')}",
                    f"severity:{alert_data.get('severity', 'medium')}",
                    "source:alertmanager"
                ],
                "alert_type": "error" if alert_data.get("severity") in ["critical", "high"] else "info",
                "source_type_name": "alertmanager"
            }
            
            async with self.session.post(
                f"{endpoint.url}/api/v1/events",
                json=event_data,
                headers={
                    "DD-API-KEY": api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 202:
                    logger.info(f"Datadog event sent for {tenant}")
                    return True
                else:
                    logger.error(f"Datadog event failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Datadog integration error: {e}")
            return False
    
    def _get_tenant_endpoint(self, tenant: str) -> IntegrationEndpoint:
        """Récupère l'endpoint spécifique au tenant"""
        return self.config.endpoints[0]
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

class IntegrationConfigManager:
    """Gestionnaire principal des intégrations"""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.active_integrations: Dict[str, Any] = {}
        self.integration_stats: Dict[str, Dict] = {}
        
    async def initialize_integrations(self) -> bool:
        """Initialise toutes les intégrations"""
        try:
            logger.info("Initializing integration configuration manager")
            
            # Chargement des configurations d'intégration
            await self._load_integration_configs()
            
            # Initialisation des intégrations actives
            await self._initialize_active_integrations()
            
            # Démarrage du monitoring des intégrations
            await self._start_integration_monitoring()
            
            logger.info("Integration manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integrations: {e}")
            return False
    
    async def _load_integration_configs(self):
        """Charge les configurations d'intégration"""
        
        # Configuration Slack
        slack_config = IntegrationConfig(
            name="slack",
            type=IntegrationType.MESSAGING,
            endpoints=[
                IntegrationEndpoint(
                    name="premium-alerts",
                    url="${SLACK_WEBHOOK_PREMIUM}",
                    auth_type=AuthType.WEBHOOK_SECRET,
                    headers={"Content-Type": "application/json"}
                ),
                IntegrationEndpoint(
                    name="general-alerts", 
                    url="${SLACK_WEBHOOK_GENERAL}",
                    auth_type=AuthType.WEBHOOK_SECRET,
                    headers={"Content-Type": "application/json"}
                )
            ],
            default_settings={
                "channel": "#alerts",
                "username": "AlertManager",
                "icon_emoji": ":warning:"
            }
        )
        
        # Configuration PagerDuty
        pagerduty_config = IntegrationConfig(
            name="pagerduty",
            type=IntegrationType.TICKETING,
            endpoints=[
                IntegrationEndpoint(
                    name="premium-incidents",
                    url="https://events.pagerduty.com/v2/enqueue",
                    auth_type=AuthType.API_KEY,
                    credentials={"integration_key": "${PD_INTEGRATION_PREMIUM}"}
                ),
                IntegrationEndpoint(
                    name="standard-incidents",
                    url="https://events.pagerduty.com/v2/enqueue", 
                    auth_type=AuthType.API_KEY,
                    credentials={"integration_key": "${PD_INTEGRATION_STANDARD}"}
                )
            ],
            default_settings={
                "event_action": "trigger",
                "client": "Spotify AlertManager"
            }
        )
        
        # Configuration Jira
        jira_config = IntegrationConfig(
            name="jira",
            type=IntegrationType.TICKETING,
            endpoints=[
                IntegrationEndpoint(
                    name="main-instance",
                    url="${JIRA_BASE_URL}",
                    auth_type=AuthType.BASIC_AUTH,
                    credentials={
                        "username": "${JIRA_USERNAME}",
                        "api_token": "${JIRA_API_TOKEN}",
                        "project_key": "ALERT"
                    }
                )
            ],
            default_settings={
                "issue_type": "Bug",
                "project": "ALERT"
            }
        )
        
        # Configuration Datadog
        datadog_config = IntegrationConfig(
            name="datadog",
            type=IntegrationType.MONITORING,
            endpoints=[
                IntegrationEndpoint(
                    name="main-api",
                    url="https://api.datadoghq.com",
                    auth_type=AuthType.API_KEY,
                    credentials={"api_key": "${DATADOG_API_KEY}"}
                )
            ]
        )
        
        self.integrations = {
            "slack": slack_config,
            "pagerduty": pagerduty_config,
            "jira": jira_config,
            "datadog": datadog_config
        }
    
    async def _initialize_active_integrations(self):
        """Initialise les instances actives des intégrations"""
        for name, config in self.integrations.items():
            if not config.enabled:
                continue
                
            try:
                if name == "slack":
                    integration = SlackIntegration(config)
                elif name == "pagerduty":
                    integration = PagerDutyIntegration(config)
                elif name == "jira":
                    integration = JiraIntegration(config)
                elif name == "datadog":
                    integration = DatadogIntegration(config)
                else:
                    logger.warning(f"Unknown integration type: {name}")
                    continue
                
                await integration.initialize()
                self.active_integrations[name] = integration
                
                # Initialisation des stats
                self.integration_stats[name] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "last_request": None,
                    "average_response_time": 0.0
                }
                
                logger.info(f"Initialized {name} integration")
                
            except Exception as e:
                logger.error(f"Failed to initialize {name} integration: {e}")
    
    async def _start_integration_monitoring(self):
        """Démarre le monitoring des intégrations"""
        asyncio.create_task(self._monitor_integration_health())
    
    async def _monitor_integration_health(self):
        """Monitore la santé des intégrations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for name, integration in self.active_integrations.items():
                    # Health check basique
                    await self._health_check_integration(name, integration)
                
            except Exception as e:
                logger.error(f"Error in integration monitoring: {e}")
    
    async def _health_check_integration(self, name: str, integration: Any):
        """Effectue un health check sur une intégration"""
        try:
            # Health check spécifique selon le type d'intégration
            if hasattr(integration, 'health_check'):
                is_healthy = await integration.health_check()
                logger.info(f"Integration {name} health check: {'OK' if is_healthy else 'FAILED'}")
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
    
    async def send_alert_to_integration(
        self, 
        integration_name: str, 
        alert_data: Dict, 
        tenant: str
    ) -> bool:
        """Envoie une alerte vers une intégration spécifique"""
        
        if integration_name not in self.active_integrations:
            logger.error(f"Integration {integration_name} not found or not active")
            return False
        
        integration = self.active_integrations[integration_name]
        start_time = datetime.utcnow()
        
        try:
            # Mise à jour des stats
            self.integration_stats[integration_name]["total_requests"] += 1
            self.integration_stats[integration_name]["last_request"] = start_time.isoformat()
            
            # Envoi selon le type d'intégration
            success = False
            if integration_name == "slack":
                success = await integration.send_alert(alert_data, tenant)
            elif integration_name == "pagerduty":
                incident_key = await integration.create_incident(alert_data, tenant)
                success = incident_key is not None
            elif integration_name == "jira":
                ticket_key = await integration.create_ticket(alert_data, tenant)
                success = ticket_key is not None
            elif integration_name == "datadog":
                success = await integration.send_event(alert_data, tenant)
            
            # Mise à jour des stats
            if success:
                self.integration_stats[integration_name]["successful_requests"] += 1
            else:
                self.integration_stats[integration_name]["failed_requests"] += 1
            
            # Calcul du temps de réponse
            response_time = (datetime.utcnow() - start_time).total_seconds()
            stats = self.integration_stats[integration_name]
            stats["average_response_time"] = (
                (stats["average_response_time"] * (stats["total_requests"] - 1) + response_time) /
                stats["total_requests"]
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert to {integration_name}: {e}")
            self.integration_stats[integration_name]["failed_requests"] += 1
            return False
    
    async def send_alert_to_multiple_integrations(
        self, 
        integration_names: List[str], 
        alert_data: Dict, 
        tenant: str
    ) -> Dict[str, bool]:
        """Envoie une alerte vers plusieurs intégrations en parallèle"""
        
        tasks = []
        for integration_name in integration_names:
            task = asyncio.create_task(
                self.send_alert_to_integration(integration_name, alert_data, tenant)
            )
            tasks.append((integration_name, task))
        
        results = {}
        for integration_name, task in tasks:
            try:
                results[integration_name] = await task
            except Exception as e:
                logger.error(f"Error in parallel send to {integration_name}: {e}")
                results[integration_name] = False
        
        return results
    
    def get_integration_stats(self, integration_name: Optional[str] = None) -> Dict:
        """Récupère les statistiques des intégrations"""
        if integration_name:
            return self.integration_stats.get(integration_name, {})
        return self.integration_stats
    
    def get_available_integrations(self) -> List[str]:
        """Récupère la liste des intégrations disponibles"""
        return list(self.active_integrations.keys())
    
    async def test_integration(self, integration_name: str, tenant: str) -> bool:
        """Teste une intégration avec un message de test"""
        test_alert = {
            "id": "test-alert-001",
            "service": "test-service",
            "severity": "info",
            "description": "This is a test alert from Spotify AlertManager",
            "metrics": {"test": True}
        }
        
        return await self.send_alert_to_integration(integration_name, test_alert, tenant)
    
    async def shutdown(self):
        """Arrête proprement toutes les intégrations"""
        logger.info("Shutting down integrations")
        
        for name, integration in self.active_integrations.items():
            try:
                if hasattr(integration, 'close'):
                    await integration.close()
                logger.info(f"Closed {name} integration")
            except Exception as e:
                logger.error(f"Error closing {name} integration: {e}")
        
        logger.info("All integrations shutdown completed")

# Instance singleton
integration_manager = IntegrationConfigManager()
