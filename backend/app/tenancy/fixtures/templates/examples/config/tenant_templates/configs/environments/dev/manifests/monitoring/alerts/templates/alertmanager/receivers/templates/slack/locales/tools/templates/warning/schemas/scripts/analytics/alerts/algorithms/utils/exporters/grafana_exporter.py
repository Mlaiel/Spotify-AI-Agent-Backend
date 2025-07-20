"""
Advanced Grafana Multi-Tenant Dashboard Exporter
===============================================

Exportateur avancé pour la création automatique de dashboards Grafana
avec isolation complète des tenants et personnalisation dynamique.

Fonctionnalités:
- Création de dashboards tenant-spécifiques
- Templates réutilisables
- Alerting intégré
- Variables dynamiques
- Annotations automatiques
- Export/Import de configurations
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urljoin
import aiohttp
import aiofiles
from jinja2 import Template, Environment, FileSystemLoader
import yaml
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GrafanaDashboardConfig:
    """Configuration pour un dashboard Grafana tenant-spécifique."""
    tenant_id: str
    dashboard_title: str
    folder_name: str = "Spotify AI Tenants"
    refresh_interval: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    tags: List[str] = field(default_factory=lambda: ["spotify-ai", "multi-tenant"])
    variables: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    shared: bool = False
    editable: bool = True


@dataclass
class PanelConfig:
    """Configuration d'un panel Grafana."""
    title: str
    panel_type: str  # graph, stat, table, heatmap, etc.
    datasource: str
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)


class GrafanaMultiTenantExporter:
    """
    Exportateur Grafana avancé avec support multi-tenant.
    
    Fonctionnalités:
    - Dashboards tenant-isolés
    - Templates réutilisables
    - Alerting automatique
    - Variables dynamiques
    - Intégration Prometheus
    """
    
    def __init__(
        self,
        grafana_url: str = "http://localhost:3000",
        api_key: str = "",
        org_id: int = 1,
        templates_path: str = "/templates/grafana"
    ):
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.org_id = org_id
        self.templates_path = templates_path
        
        # Configuration HTTP
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'X-Grafana-Org-Id': str(org_id)
        }
        
        # Templates Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_path),
            autoescape=True
        )
        
        # Cache des dashboards créés
        self.created_dashboards: Dict[str, Dict[str, Any]] = {}
        
        # Métriques internes
        self._setup_internal_tracking()
        
    def _setup_internal_tracking(self):
        """Configure le suivi interne des opérations."""
        self.stats = {
            'dashboards_created': 0,
            'dashboards_updated': 0,
            'alerts_configured': 0,
            'folders_created': 0,
            'last_operation': None
        }
        
    async def initialize(self):
        """Initialise l'exportateur Grafana."""
        try:
            # Vérifier la connexion Grafana
            await self._check_grafana_connection()
            
            # Créer le dossier principal pour les tenants
            await self._ensure_tenant_folder()
            
            # Charger les templates
            await self._load_dashboard_templates()
            
            logger.info("GrafanaMultiTenantExporter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Grafana exporter: {e}")
            raise
            
    async def _check_grafana_connection(self):
        """Vérifie la connexion à Grafana."""
        url = urljoin(self.grafana_url, '/api/health')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Grafana connection failed: {response.status}")
                    
                health_data = await response.json()
                if health_data.get('database') != 'ok':
                    raise Exception("Grafana database not healthy")
                    
    async def _ensure_tenant_folder(self):
        """S'assure que le dossier pour les tenants existe."""
        folder_data = {
            'title': 'Spotify AI Tenants',
            'uid': 'spotify-ai-tenants'
        }
        
        url = urljoin(self.grafana_url, '/api/folders')
        
        async with aiohttp.ClientSession() as session:
            # Vérifier si le dossier existe
            folder_url = f"{url}/spotify-ai-tenants"
            async with session.get(folder_url, headers=self.headers) as response:
                if response.status == 404:
                    # Créer le dossier
                    async with session.post(
                        url, 
                        json=folder_data, 
                        headers=self.headers
                    ) as create_response:
                        if create_response.status in [200, 201]:
                            self.stats['folders_created'] += 1
                            logger.info("Created Spotify AI Tenants folder")
                        else:
                            logger.error(
                                f"Failed to create folder: {create_response.status}"
                            )
                            
    async def _load_dashboard_templates(self):
        """Charge les templates de dashboards."""
        self.templates = {
            'ai_metrics': 'spotify_ai_metrics_dashboard.json',
            'business_metrics': 'spotify_business_dashboard.json',
            'collaboration_metrics': 'spotify_collaboration_dashboard.json',
            'performance_overview': 'spotify_performance_overview.json'
        }
        
    async def create_tenant_dashboard(
        self,
        config: GrafanaDashboardConfig,
        dashboard_type: str = 'ai_metrics'
    ) -> Dict[str, Any]:
        """
        Crée un dashboard Grafana pour un tenant spécifique.
        
        Args:
            config: Configuration du dashboard
            dashboard_type: Type de dashboard à créer
            
        Returns:
            Informations sur le dashboard créé
        """
        try:
            # Générer le dashboard à partir du template
            dashboard_json = await self._generate_dashboard_from_template(
                config, dashboard_type
            )
            
            # Créer le dashboard dans Grafana
            result = await self._create_dashboard_in_grafana(dashboard_json)
            
            # Configurer les alertes
            await self._setup_dashboard_alerts(config, result['dashboard']['uid'])
            
            # Sauvegarder la configuration
            self.created_dashboards[config.tenant_id] = {
                'uid': result['dashboard']['uid'],
                'url': result['url'],
                'type': dashboard_type,
                'created_at': datetime.now().isoformat(),
                'config': config
            }
            
            self.stats['dashboards_created'] += 1
            self.stats['last_operation'] = datetime.now().isoformat()
            
            logger.info(
                "Created tenant dashboard",
                tenant_id=config.tenant_id,
                dashboard_uid=result['dashboard']['uid'],
                dashboard_type=dashboard_type
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to create tenant dashboard: {e}",
                tenant_id=config.tenant_id,
                dashboard_type=dashboard_type
            )
            raise
            
    async def _generate_dashboard_from_template(
        self,
        config: GrafanaDashboardConfig,
        dashboard_type: str
    ) -> Dict[str, Any]:
        """Génère un dashboard à partir d'un template."""
        
        # Dashboard de base pour les métriques IA Spotify
        if dashboard_type == 'ai_metrics':
            return await self._create_ai_metrics_dashboard(config)
        elif dashboard_type == 'business_metrics':
            return await self._create_business_metrics_dashboard(config)
        elif dashboard_type == 'collaboration_metrics':
            return await self._create_collaboration_dashboard(config)
        elif dashboard_type == 'performance_overview':
            return await self._create_performance_overview_dashboard(config)
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
            
    async def _create_ai_metrics_dashboard(
        self, 
        config: GrafanaDashboardConfig
    ) -> Dict[str, Any]:
        """Crée un dashboard pour les métriques IA."""
        
        dashboard = {
            'dashboard': {
                'id': None,
                'title': f"{config.dashboard_title} - AI Metrics",
                'tags': config.tags + ['ai-metrics'],
                'timezone': 'UTC',
                'refresh': config.refresh_interval,
                'time': {
                    'from': config.time_from,
                    'to': config.time_to
                },
                'templating': {
                    'list': await self._create_dashboard_variables(config)
                },
                'annotations': {
                    'list': await self._create_dashboard_annotations(config)
                },
                'panels': [
                    # Panel 1: Temps d'inférence IA
                    {
                        'id': 1,
                        'title': 'AI Model Inference Time',
                        'type': 'timeseries',
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0},
                        'targets': [{
                            'expr': f'spotify_ai_ai_inference_duration_seconds{{tenant_id="{config.tenant_id}"}}',
                            'refId': 'A',
                            'legendFormat': '{{model_name}} - {{model_version}}'
                        }],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 's',
                                'min': 0,
                                'thresholds': {
                                    'steps': [
                                        {'color': 'green', 'value': None},
                                        {'color': 'yellow', 'value': 0.1},
                                        {'color': 'red', 'value': 0.5}
                                    ]
                                }
                            }
                        },
                        'options': {
                            'tooltip': {'mode': 'multi'},
                            'legend': {'displayMode': 'table', 'placement': 'bottom'}
                        }
                    },
                    
                    # Panel 2: Précision des recommandations
                    {
                        'id': 2,
                        'title': 'Recommendation Accuracy',
                        'type': 'stat',
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0},
                        'targets': [{
                            'expr': f'avg(spotify_ai_recommendation_accuracy_ratio{{tenant_id="{config.tenant_id}"}})',
                            'refId': 'A'
                        }],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 'percentunit',
                                'min': 0,
                                'max': 1,
                                'thresholds': {
                                    'steps': [
                                        {'color': 'red', 'value': None},
                                        {'color': 'yellow', 'value': 0.7},
                                        {'color': 'green', 'value': 0.9}
                                    ]
                                }
                            }
                        },
                        'options': {
                            'colorMode': 'background',
                            'graphMode': 'area',
                            'justifyMode': 'center'
                        }
                    },
                    
                    # Panel 3: Score d'engagement des artistes
                    {
                        'id': 3,
                        'title': 'Artist Engagement Score',
                        'type': 'gauge',
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8},
                        'targets': [{
                            'expr': f'spotify_ai_artist_engagement_score{{tenant_id="{config.tenant_id}"}}',
                            'refId': 'A',
                            'legendFormat': '{{artist_tier}} - {{region}}'
                        }],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 'none',
                                'min': 0,
                                'max': 10,
                                'thresholds': {
                                    'steps': [
                                        {'color': 'red', 'value': None},
                                        {'color': 'yellow', 'value': 5},
                                        {'color': 'green', 'value': 8}
                                    ]
                                }
                            }
                        },
                        'options': {
                            'showThresholdLabels': True,
                            'showThresholdMarkers': True
                        }
                    },
                    
                    # Panel 4: Heatmap des performances par modèle
                    {
                        'id': 4,
                        'title': 'Model Performance Heatmap',
                        'type': 'heatmap',
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8},
                        'targets': [{
                            'expr': f'rate(spotify_ai_ai_inference_duration_seconds_count{{tenant_id="{config.tenant_id}"}}[5m])',
                            'refId': 'A',
                            'format': 'time_series'
                        }],
                        'options': {
                            'calculate': True,
                            'yAxis': {
                                'unit': 's',
                                'min': 'auto',
                                'max': 'auto'
                            }
                        }
                    }
                ],
                'editable': config.editable,
                'fiscalYearStartMonth': 0,
                'graphTooltip': 1,
                'links': [],
                'liveNow': False,
                'schemaVersion': 37,
                'style': 'dark',
                'uid': f"spotify-ai-{config.tenant_id}-ai-metrics",
                'version': 1,
                'weekStart': ''
            },
            'folderId': await self._get_folder_id('Spotify AI Tenants'),
            'overwrite': True
        }
        
        return dashboard
        
    async def _create_business_metrics_dashboard(
        self, 
        config: GrafanaDashboardConfig
    ) -> Dict[str, Any]:
        """Crée un dashboard pour les métriques business."""
        
        dashboard = {
            'dashboard': {
                'id': None,
                'title': f"{config.dashboard_title} - Business Metrics",
                'tags': config.tags + ['business-metrics'],
                'timezone': 'UTC',
                'refresh': config.refresh_interval,
                'time': {
                    'from': config.time_from,
                    'to': config.time_to
                },
                'templating': {
                    'list': await self._create_dashboard_variables(config)
                },
                'panels': [
                    # Panel 1: Tracks générés
                    {
                        'id': 1,
                        'title': 'Tracks Generated',
                        'type': 'timeseries',
                        'gridPos': {'h': 8, 'w': 8, 'x': 0, 'y': 0},
                        'targets': [{
                            'expr': f'increase(spotify_ai_tracks_generated_total{{tenant_id="{config.tenant_id}"}}[1h])',
                            'refId': 'A',
                            'legendFormat': '{{genre}} - {{collaboration_type}}'
                        }],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 'short',
                                'min': 0
                            }
                        }
                    },
                    
                    # Panel 2: Impact revenue
                    {
                        'id': 2,
                        'title': 'Revenue Impact (€)',
                        'type': 'stat',
                        'gridPos': {'h': 8, 'w': 8, 'x': 8, 'y': 0},
                        'targets': [{
                            'expr': f'sum(spotify_ai_revenue_impact_euros{{tenant_id="{config.tenant_id}"}})',
                            'refId': 'A'
                        }],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 'currencyEUR',
                                'decimals': 2,
                                'color': {'mode': 'thresholds'},
                                'thresholds': {
                                    'steps': [
                                        {'color': 'red', 'value': None},
                                        {'color': 'yellow', 'value': 10000},
                                        {'color': 'green', 'value': 50000}
                                    ]
                                }
                            }
                        },
                        'options': {
                            'colorMode': 'background',
                            'graphMode': 'area'
                        }
                    },
                    
                    # Panel 3: Succès des collaborations
                    {
                        'id': 3,
                        'title': 'Collaboration Success Rate',
                        'type': 'piechart',
                        'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 0},
                        'targets': [{
                            'expr': f'avg_over_time(spotify_ai_collaboration_success_rate{{tenant_id="{config.tenant_id}"}}[1h])',
                            'refId': 'A',
                            'legendFormat': '{{collaboration_type}}'
                        }],
                        'options': {
                            'reduceOptions': {
                                'values': False,
                                'calcs': ['lastNotNull'],
                                'fields': ''
                            },
                            'pieType': 'pie',
                            'tooltip': {'mode': 'single'},
                            'legend': {'displayMode': 'visible', 'placement': 'bottom'}
                        }
                    }
                ],
                'editable': config.editable,
                'uid': f"spotify-ai-{config.tenant_id}-business",
                'version': 1
            },
            'folderId': await self._get_folder_id('Spotify AI Tenants'),
            'overwrite': True
        }
        
        return dashboard
        
    async def _create_collaboration_dashboard(
        self, 
        config: GrafanaDashboardConfig
    ) -> Dict[str, Any]:
        """Crée un dashboard pour les métriques de collaboration."""
        
        dashboard = {
            'dashboard': {
                'id': None,
                'title': f"{config.dashboard_title} - Collaboration Analytics",
                'tags': config.tags + ['collaboration'],
                'timezone': 'UTC',
                'refresh': config.refresh_interval,
                'panels': [
                    # Network graph des collaborations
                    {
                        'id': 1,
                        'title': 'Artist Collaboration Network',
                        'type': 'nodeGraph',
                        'gridPos': {'h': 12, 'w': 24, 'x': 0, 'y': 0},
                        'targets': [{
                            'expr': f'spotify_ai_collaboration_success_rate{{tenant_id="{config.tenant_id}"}}',
                            'refId': 'A'
                        }],
                        'options': {
                            'nodes': {
                                'mainStatUnit': 'none',
                                'arcColorField': 'success_rate',
                                'colorField': 'genre_match'
                            },
                            'edges': {
                                'mainStatUnit': 'none',
                                'colorField': 'collaboration_strength'
                            }
                        }
                    }
                ],
                'editable': config.editable,
                'uid': f"spotify-ai-{config.tenant_id}-collaboration",
                'version': 1
            },
            'folderId': await self._get_folder_id('Spotify AI Tenants'),
            'overwrite': True
        }
        
        return dashboard
        
    async def _create_performance_overview_dashboard(
        self, 
        config: GrafanaDashboardConfig
    ) -> Dict[str, Any]:
        """Crée un dashboard de vue d'ensemble des performances."""
        
        dashboard = {
            'dashboard': {
                'id': None,
                'title': f"{config.dashboard_title} - Performance Overview",
                'tags': config.tags + ['overview'],
                'timezone': 'UTC',
                'refresh': config.refresh_interval,
                'panels': [
                    # Row pour les métriques système
                    {
                        'id': 1,
                        'title': 'System Performance',
                        'type': 'row',
                        'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 0},
                        'collapsed': False,
                        'panels': []
                    },
                    
                    # CPU et mémoire
                    {
                        'id': 2,
                        'title': 'Resource Usage',
                        'type': 'timeseries',
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 1},
                        'targets': [
                            {
                                'expr': f'rate(cpu_usage_seconds_total{{tenant_id="{config.tenant_id}"}}[5m]) * 100',
                                'refId': 'A',
                                'legendFormat': 'CPU %'
                            },
                            {
                                'expr': f'(memory_usage_bytes{{tenant_id="{config.tenant_id}"}}/memory_total_bytes{{tenant_id="{config.tenant_id}"}}) * 100',
                                'refId': 'B',
                                'legendFormat': 'Memory %'
                            }
                        ],
                        'fieldConfig': {
                            'defaults': {
                                'unit': 'percent',
                                'min': 0,
                                'max': 100
                            }
                        }
                    }
                ],
                'editable': config.editable,
                'uid': f"spotify-ai-{config.tenant_id}-overview",
                'version': 1
            },
            'folderId': await self._get_folder_id('Spotify AI Tenants'),
            'overwrite': True
        }
        
        return dashboard
        
    async def _create_dashboard_variables(
        self, 
        config: GrafanaDashboardConfig
    ) -> List[Dict[str, Any]]:
        """Crée les variables pour le dashboard."""
        
        variables = [
            # Variable pour le tenant (fixe)
            {
                'name': 'tenant_id',
                'type': 'constant',
                'current': {
                    'value': config.tenant_id,
                    'text': config.tenant_id
                },
                'hide': 2,  # Caché
                'query': config.tenant_id
            },
            
            # Variable pour l'intervalle de temps
            {
                'name': 'interval',
                'type': 'interval',
                'current': {
                    'value': '5m',
                    'text': '5m'
                },
                'options': [
                    {'text': '1m', 'value': '1m'},
                    {'text': '5m', 'value': '5m', 'selected': True},
                    {'text': '15m', 'value': '15m'},
                    {'text': '1h', 'value': '1h'}
                ],
                'query': '1m,5m,15m,1h',
                'auto': True,
                'auto_count': 10,
                'auto_min': '1m'
            },
            
            # Variable pour les modèles IA
            {
                'name': 'model_name',
                'type': 'query',
                'datasource': 'Prometheus',
                'query': f'label_values(spotify_ai_ai_inference_duration_seconds{{tenant_id="{config.tenant_id}"}}, model_name)',
                'current': {
                    'value': 'All',
                    'text': 'All'
                },
                'multi': True,
                'includeAll': True,
                'allValue': '.*'
            }
        ]
        
        # Ajouter les variables personnalisées
        for var_name, var_config in config.variables.items():
            variables.append({
                'name': var_name,
                **var_config
            })
            
        return variables
        
    async def _create_dashboard_annotations(
        self, 
        config: GrafanaDashboardConfig
    ) -> List[Dict[str, Any]]:
        """Crée les annotations pour le dashboard."""
        
        annotations = [
            # Annotations des déploiements
            {
                'name': 'Deployments',
                'datasource': 'Prometheus',
                'enable': True,
                'hide': False,
                'iconColor': 'blue',
                'query': f'changes(spotify_ai_deployment_info{{tenant_id="{config.tenant_id}"}}[1m])',
                'step': '60s',
                'tagKeys': 'version,environment',
                'textFormat': 'Deployment: {{version}}',
                'titleFormat': 'New Deployment'
            },
            
            # Annotations des incidents
            {
                'name': 'Incidents',
                'datasource': 'Prometheus',
                'enable': True,
                'hide': False,
                'iconColor': 'red',
                'query': f'spotify_ai_incident_active{{tenant_id="{config.tenant_id}"}}',
                'step': '60s',
                'tagKeys': 'severity,component',
                'textFormat': 'Incident: {{component}} - {{severity}}',
                'titleFormat': 'Active Incident'
            }
        ]
        
        # Ajouter les annotations personnalisées
        annotations.extend(config.annotations)
        
        return annotations
        
    async def _get_folder_id(self, folder_title: str) -> Optional[int]:
        """Récupère l'ID d'un dossier Grafana."""
        url = urljoin(self.grafana_url, '/api/folders')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    folders = await response.json()
                    for folder in folders:
                        if folder['title'] == folder_title:
                            return folder['id']
        return None
        
    async def _create_dashboard_in_grafana(
        self, 
        dashboard_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crée le dashboard dans Grafana."""
        url = urljoin(self.grafana_url, '/api/dashboards/db')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=dashboard_json, 
                headers=self.headers
            ) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to create dashboard: {response.status} - {error_text}"
                    )
                    
    async def _setup_dashboard_alerts(
        self, 
        config: GrafanaDashboardConfig, 
        dashboard_uid: str
    ):
        """Configure les alertes pour le dashboard."""
        
        alerts = [
            # Alerte sur le temps d'inférence élevé
            {
                'title': f'High AI Inference Time - {config.tenant_id}',
                'message': 'AI model inference time is above threshold',
                'frequency': '1m',
                'conditions': [{
                    'query': {
                        'queryType': '',
                        'refId': 'A',
                        'datasourceUid': 'prometheus',
                        'model': {
                            'expr': f'avg(spotify_ai_ai_inference_duration_seconds{{tenant_id="{config.tenant_id}"}}) > 0.5',
                            'intervalMs': 1000,
                            'maxDataPoints': 43200
                        }
                    },
                    'reducer': {
                        'type': 'last',
                        'params': []
                    },
                    'evaluator': {
                        'params': [0.5],
                        'type': 'gt'
                    }
                }],
                'executionErrorState': 'alerting',
                'noDataState': 'no_data',
                'for': '5m',
                'annotations': {
                    'description': 'AI inference time is consistently above 500ms',
                    'runbook_url': 'https://docs.spotify-ai.com/runbooks/high-inference-time',
                    'summary': 'High AI inference latency detected'
                },
                'labels': {
                    'tenant_id': config.tenant_id,
                    'severity': 'warning',
                    'component': 'ai-inference'
                }
            },
            
            # Alerte sur la précision faible
            {
                'title': f'Low Recommendation Accuracy - {config.tenant_id}',
                'message': 'Recommendation accuracy dropped below acceptable threshold',
                'frequency': '5m',
                'conditions': [{
                    'query': {
                        'queryType': '',
                        'refId': 'A',
                        'datasourceUid': 'prometheus',
                        'model': {
                            'expr': f'avg(spotify_ai_recommendation_accuracy_ratio{{tenant_id="{config.tenant_id}"}}) < 0.8',
                            'intervalMs': 1000,
                            'maxDataPoints': 43200
                        }
                    },
                    'reducer': {
                        'type': 'last',
                        'params': []
                    },
                    'evaluator': {
                        'params': [0.8],
                        'type': 'lt'
                    }
                }],
                'executionErrorState': 'alerting',
                'noDataState': 'no_data',
                'for': '10m',
                'annotations': {
                    'description': 'Recommendation accuracy below 80%',
                    'runbook_url': 'https://docs.spotify-ai.com/runbooks/low-accuracy',
                    'summary': 'Poor recommendation performance detected'
                },
                'labels': {
                    'tenant_id': config.tenant_id,
                    'severity': 'critical',
                    'component': 'recommendations'
                }
            }
        ]
        
        # Créer les alertes dans Grafana
        for alert_config in alerts:
            await self._create_alert_rule(alert_config)
            
        self.stats['alerts_configured'] += len(alerts)
        
    async def _create_alert_rule(self, alert_config: Dict[str, Any]):
        """Crée une règle d'alerte dans Grafana."""
        url = urljoin(self.grafana_url, '/api/ruler/grafana/api/v1/rules/spotify-ai-alerts')
        
        rule_group = {
            'name': 'spotify-ai-alerts',
            'interval': '1m',
            'rules': [alert_config]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=rule_group, 
                headers=self.headers
            ) as response:
                if response.status not in [200, 201, 202]:
                    logger.warning(
                        f"Failed to create alert rule: {response.status}",
                        alert_title=alert_config['title']
                    )
                    
    async def update_tenant_dashboard(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Met à jour un dashboard existant."""
        if tenant_id not in self.created_dashboards:
            raise ValueError(f"No dashboard found for tenant: {tenant_id}")
            
        dashboard_info = self.created_dashboards[tenant_id]
        dashboard_uid = dashboard_info['uid']
        
        # Récupérer le dashboard actuel
        url = urljoin(self.grafana_url, f'/api/dashboards/uid/{dashboard_uid}')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    current_dashboard = await response.json()
                    
                    # Appliquer les mises à jour
                    dashboard_data = current_dashboard['dashboard']
                    for key, value in updates.items():
                        dashboard_data[key] = value
                        
                    # Incrémenter la version
                    dashboard_data['version'] += 1
                    
                    # Sauvegarder
                    update_payload = {
                        'dashboard': dashboard_data,
                        'overwrite': True
                    }
                    
                    async with session.post(
                        urljoin(self.grafana_url, '/api/dashboards/db'),
                        json=update_payload,
                        headers=self.headers
                    ) as update_response:
                        if update_response.status in [200, 201]:
                            self.stats['dashboards_updated'] += 1
                            return await update_response.json()
                        else:
                            raise Exception(
                                f"Failed to update dashboard: {update_response.status}"
                            )
                else:
                    raise Exception(f"Dashboard not found: {dashboard_uid}")
                    
    async def export_tenant_dashboards(
        self, 
        tenant_ids: Optional[List[str]] = None,
        export_path: str = "/exports/dashboards"
    ):
        """Exporte les dashboards de tenants."""
        if tenant_ids is None:
            tenant_ids = list(self.created_dashboards.keys())
            
        exported_dashboards = {}
        
        for tenant_id in tenant_ids:
            if tenant_id in self.created_dashboards:
                dashboard_info = self.created_dashboards[tenant_id]
                dashboard_uid = dashboard_info['uid']
                
                # Récupérer le dashboard
                url = urljoin(self.grafana_url, f'/api/dashboards/uid/{dashboard_uid}')
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 200:
                            dashboard_data = await response.json()
                            
                            # Sauvegarder dans un fichier
                            filename = f"{export_path}/dashboard_{tenant_id}_{dashboard_uid}.json"
                            async with aiofiles.open(filename, 'w') as f:
                                await f.write(json.dumps(dashboard_data, indent=2))
                                
                            exported_dashboards[tenant_id] = {
                                'file': filename,
                                'uid': dashboard_uid,
                                'exported_at': datetime.now().isoformat()
                            }
                            
        return exported_dashboards
        
    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des dashboards."""
        return {
            'stats': self.stats,
            'created_dashboards': len(self.created_dashboards),
            'tenant_list': list(self.created_dashboards.keys()),
            'last_operation': self.stats['last_operation']
        }
        
    async def cleanup(self):
        """Nettoie les ressources."""
        logger.info("GrafanaMultiTenantExporter cleaned up")


# Factory pour créer des exportateurs Grafana
class GrafanaExporterFactory:
    """Factory pour créer des exportateurs Grafana configurés."""
    
    @staticmethod
    def create_spotify_ai_exporter(
        grafana_url: str,
        api_key: str,
        org_id: int = 1
    ) -> GrafanaMultiTenantExporter:
        """Crée un exportateur configuré pour Spotify AI."""
        return GrafanaMultiTenantExporter(
            grafana_url=grafana_url,
            api_key=api_key,
            org_id=org_id,
            templates_path="/templates/grafana"
        )


# Usage example
if __name__ == "__main__":
    async def main():
        # Configuration Grafana
        exporter = GrafanaExporterFactory.create_spotify_ai_exporter(
            grafana_url="http://localhost:3000",
            api_key="eyJrIjoiT0tTcG1pUlY2RnVKZTFVaDFsNFZXdE9ZWmNrMkZYbk"
        )
        
        await exporter.initialize()
        
        # Configuration pour un artiste
        config = GrafanaDashboardConfig(
            tenant_id="spotify_artist_daft_punk",
            dashboard_title="Daft Punk AI Analytics",
            tags=["daft-punk", "electronic", "ai-analytics"],
            variables={
                'genre': {
                    'type': 'constant',
                    'query': 'electronic',
                    'current': {'value': 'electronic', 'text': 'Electronic'}
                }
            }
        )
        
        # Créer les dashboards
        ai_dashboard = await exporter.create_tenant_dashboard(config, 'ai_metrics')
        business_dashboard = await exporter.create_tenant_dashboard(config, 'business_metrics')
        
        print(f"Created AI dashboard: {ai_dashboard['url']}")
        print(f"Created business dashboard: {business_dashboard['url']}")
        
        # Statistiques
        stats = await exporter.get_dashboard_stats()
        print(f"Dashboard stats: {stats}")
        
        await exporter.cleanup()
        
    asyncio.run(main())
