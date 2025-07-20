"""
Générateur de Tableaux de Bord Automatique - Spotify AI Agent
=============================================================

Système intelligent de génération automatique de dashboards Grafana
avec personnalisation par tenant et adaptation dynamique.

Fonctionnalités:
- Génération automatique de dashboards Grafana
- Personnalisation par tenant et service
- Templates adaptatifs basés sur les métriques disponibles
- Intégration avec Prometheus, InfluxDB, TimescaleDB
- Alertes visuelles et annotations automatiques
- Export/Import de configurations de dashboard
- Analytics et optimisation des performances
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import jinja2
from pathlib import Path
import aiohttp
import redis.asyncio as redis


class DashboardType(Enum):
    """Types de tableaux de bord"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    SECURITY = "security"
    BUSINESS = "business"
    ML_ANALYTICS = "ml_analytics"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class VisualizationType(Enum):
    """Types de visualisations"""
    TIME_SERIES = "timeseries"
    SINGLE_STAT = "singlestat"
    TABLE = "table"
    HEATMAP = "heatmap"
    PIE_CHART = "piechart"
    BAR_GAUGE = "bargauge"
    GAUGE = "gauge"
    TEXT = "text"
    GRAPH = "graph"


@dataclass
class Panel:
    """Panel de dashboard"""
    id: int
    title: str
    type: VisualizationType
    datasource: str
    targets: List[Dict[str, Any]] = field(default_factory=list)
    grid_pos: Dict[str, int] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    alert: Optional[Dict[str, Any]] = None
    thresholds: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Dashboard:
    """Configuration complète d'un dashboard"""
    id: Optional[int] = None
    uid: str = ""
    title: str = ""
    description: str = ""
    dashboard_type: DashboardType = DashboardType.TECHNICAL
    tenant_id: str = ""
    service: str = ""
    environment: str = ""
    panels: List[Panel] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    time_range: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    refresh_interval: str = "30s"
    variables: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    auto_generated: bool = True
    template_version: str = "1.0"


class DashboardGenerator:
    """Générateur intelligent de tableaux de bord"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration Grafana
        self.grafana_url = config.get('grafana_url', 'http://localhost:3000')
        self.grafana_api_key = config.get('grafana_api_key')
        self.grafana_org_id = config.get('grafana_org_id', 1)
        
        # Configuration des datasources
        self.default_datasource = config.get('default_datasource', 'Prometheus')
        self.datasources = config.get('datasources', {})
        
        # Templates de dashboards
        self.dashboard_templates: Dict[DashboardType, Dict[str, Any]] = {}
        self.panel_templates: Dict[str, Dict[str, Any]] = {}
        
        # Cache
        self.redis_client = None
        self.generated_dashboards: Dict[str, Dashboard] = {}
        
        # Moteur de templates Jinja2
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({}),
            autoescape=True
        )
        
        # Session HTTP pour les API calls
        self.http_session = None
        
        # Métriques disponibles par tenant/service
        self.available_metrics: Dict[str, List[str]] = {}
        
        # Configuration des couleurs et thèmes
        self.color_schemes = {
            'default': ['#7EB26D', '#EAB839', '#6ED0E0', '#EF843C', '#E24D42'],
            'spotify': ['#1DB954', '#191414', '#FFFFFF', '#535353', '#B3B3B3'],
            'performance': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'security': ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB']
        }
        
    async def initialize(self):
        """Initialisation asynchrone du générateur"""
        try:
            # Connexion Redis
            if self.config.get('redis_url'):
                self.redis_client = redis.from_url(
                    self.config['redis_url'],
                    decode_responses=True
                )
            
            # Session HTTP
            self.http_session = aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {self.grafana_api_key}'},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Chargement des templates
            await self._load_dashboard_templates()
            await self._load_panel_templates()
            
            # Découverte des métriques disponibles
            await self._discover_available_metrics()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._periodic_dashboard_update())
            asyncio.create_task(self._cleanup_old_dashboards())
            
            self.logger.info("DashboardGenerator initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def generate_tenant_dashboard(self, 
                                      tenant_id: str,
                                      dashboard_type: DashboardType = DashboardType.TECHNICAL,
                                      custom_config: Dict[str, Any] = None) -> Dashboard:
        """Génération d'un dashboard pour un tenant"""
        try:
            # Configuration par défaut
            config = custom_config or {}
            
            # Recherche des métriques disponibles pour ce tenant
            tenant_metrics = await self._get_tenant_metrics(tenant_id)
            
            # Sélection du template approprié
            template = self.dashboard_templates.get(dashboard_type, {})
            
            # Création du dashboard
            dashboard = Dashboard(
                uid=f"{tenant_id}_{dashboard_type.value}_{int(datetime.utcnow().timestamp())}",
                title=f"{tenant_id.title()} - {dashboard_type.value.title()} Dashboard",
                description=f"Dashboard automatique pour {tenant_id}",
                dashboard_type=dashboard_type,
                tenant_id=tenant_id,
                tags=[tenant_id, dashboard_type.value, "auto-generated"],
                refresh_interval=config.get('refresh_interval', '30s')
            )
            
            # Génération des panels
            panels = await self._generate_panels_for_tenant(
                tenant_id, dashboard_type, tenant_metrics, config
            )
            dashboard.panels = panels
            
            # Génération des variables de dashboard
            variables = await self._generate_dashboard_variables(tenant_id, tenant_metrics)
            dashboard.variables = variables
            
            # Génération des annotations
            annotations = await self._generate_dashboard_annotations(tenant_id)
            dashboard.annotations = annotations
            
            # Sauvegarde en cache
            cache_key = f"dashboard:{tenant_id}:{dashboard_type.value}"
            self.generated_dashboards[cache_key] = dashboard
            
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 heure de cache
                    json.dumps(dashboard.__dict__, default=str)
                )
            
            self.logger.info(f"Dashboard généré pour {tenant_id}: {dashboard.uid}")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de dashboard: {e}")
            raise
    
    async def deploy_dashboard_to_grafana(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Déploiement d'un dashboard vers Grafana"""
        try:
            # Conversion au format Grafana
            grafana_dashboard = await self._convert_to_grafana_format(dashboard)
            
            # Payload pour l'API Grafana
            payload = {
                "dashboard": grafana_dashboard,
                "overwrite": True,
                "message": f"Auto-generated dashboard for {dashboard.tenant_id}"
            }
            
            # Appel à l'API Grafana
            url = f"{self.grafana_url}/api/dashboards/db"
            async with self.http_session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    dashboard.id = result.get('id')
                    dashboard.uid = result.get('uid', dashboard.uid)
                    
                    self.logger.info(f"Dashboard déployé: {dashboard.uid}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Erreur Grafana API: {response.status} - {error_text}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors du déploiement: {e}")
            raise
    
    async def update_existing_dashboard(self, dashboard_uid: str, updates: Dict[str, Any]) -> bool:
        """Mise à jour d'un dashboard existant"""
        try:
            # Récupération du dashboard existant
            existing_dashboard = await self._get_dashboard_from_grafana(dashboard_uid)
            if not existing_dashboard:
                return False
            
            # Application des mises à jour
            for key, value in updates.items():
                if hasattr(existing_dashboard, key):
                    setattr(existing_dashboard, key, value)
            
            existing_dashboard.updated_at = datetime.utcnow()
            
            # Redéploiement
            await self.deploy_dashboard_to_grafana(existing_dashboard)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour: {e}")
            return False
    
    async def generate_service_dashboard(self,
                                       tenant_id: str,
                                       service: str,
                                       dashboard_type: DashboardType = DashboardType.PERFORMANCE) -> Dashboard:
        """Génération d'un dashboard spécifique à un service"""
        try:
            # Métriques spécifiques au service
            service_metrics = await self._get_service_metrics(tenant_id, service)
            
            dashboard = Dashboard(
                uid=f"{tenant_id}_{service}_{dashboard_type.value}_{int(datetime.utcnow().timestamp())}",
                title=f"{service.title()} Service - {dashboard_type.value.title()}",
                description=f"Dashboard automatique pour le service {service}",
                dashboard_type=dashboard_type,
                tenant_id=tenant_id,
                service=service,
                tags=[tenant_id, service, dashboard_type.value, "service-specific"]
            )
            
            # Panels spécifiques au service
            panels = await self._generate_service_panels(
                tenant_id, service, dashboard_type, service_metrics
            )
            dashboard.panels = panels
            
            # Variables spécifiques
            variables = await self._generate_service_variables(tenant_id, service)
            dashboard.variables = variables
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de dashboard service: {e}")
            raise
    
    async def create_custom_panel(self,
                                panel_config: Dict[str, Any],
                                metrics: List[str],
                                tenant_id: str) -> Panel:
        """Création d'un panel personnalisé"""
        try:
            panel = Panel(
                id=panel_config.get('id', len(self.panel_templates)),
                title=panel_config['title'],
                type=VisualizationType(panel_config.get('type', 'timeseries')),
                datasource=panel_config.get('datasource', self.default_datasource),
                grid_pos=panel_config.get('grid_pos', {'h': 8, 'w': 12, 'x': 0, 'y': 0})
            )
            
            # Génération des requêtes
            targets = []
            for i, metric in enumerate(metrics):
                target = {
                    'expr': f'{metric}{{tenant_id="{tenant_id}"}}',
                    'refId': chr(65 + i),  # A, B, C, etc.
                    'legendFormat': f'{metric}',
                    'interval': '30s'
                }
                targets.append(target)
            
            panel.targets = targets
            
            # Configuration des options selon le type
            panel.options = await self._get_panel_options(panel.type, panel_config)
            panel.field_config = await self._get_field_config(panel.type, panel_config)
            
            # Seuils et alertes
            if 'thresholds' in panel_config:
                panel.thresholds = panel_config['thresholds']
            
            if 'alert' in panel_config:
                panel.alert = await self._create_panel_alert(panel_config['alert'], metrics)
            
            return panel
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de panel: {e}")
            raise
    
    async def _generate_panels_for_tenant(self,
                                        tenant_id: str,
                                        dashboard_type: DashboardType,
                                        metrics: List[str],
                                        config: Dict[str, Any]) -> List[Panel]:
        """Génération des panels pour un tenant"""
        panels = []
        panel_id = 1
        y_position = 0
        
        try:
            if dashboard_type == DashboardType.EXECUTIVE:
                # Dashboard exécutif - métriques business et SLA
                executive_panels = [
                    {
                        'title': 'Service Availability',
                        'metrics': ['up', 'http_requests_total'],
                        'type': 'gauge'
                    },
                    {
                        'title': 'Response Time Trends',
                        'metrics': ['http_request_duration_seconds'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Error Rate',
                        'metrics': ['http_requests_total'],
                        'type': 'singlestat'
                    },
                    {
                        'title': 'Active Users',
                        'metrics': ['spotify_active_users'],
                        'type': 'timeseries'
                    }
                ]
                
                for panel_config in executive_panels:
                    available_metrics = [m for m in panel_config['metrics'] if m in metrics]
                    if available_metrics:
                        panel = await self._create_standard_panel(
                            panel_id, panel_config, available_metrics, tenant_id, y_position
                        )
                        panels.append(panel)
                        panel_id += 1
                        y_position += 8
            
            elif dashboard_type == DashboardType.TECHNICAL:
                # Dashboard technique - métriques système et performance
                technical_panels = [
                    {
                        'title': 'CPU Usage',
                        'metrics': ['system_cpu_percent'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Memory Usage',
                        'metrics': ['system_memory_percent'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'API Request Rate',
                        'metrics': ['http_requests_total'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Database Connections',
                        'metrics': ['database_connections_active'],
                        'type': 'gauge'
                    },
                    {
                        'title': 'Queue Length',
                        'metrics': ['queue_length'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Cache Hit Rate',
                        'metrics': ['cache_hit_rate'],
                        'type': 'singlestat'
                    }
                ]
                
                for panel_config in technical_panels:
                    available_metrics = [m for m in panel_config['metrics'] if m in metrics]
                    if available_metrics:
                        panel = await self._create_standard_panel(
                            panel_id, panel_config, available_metrics, tenant_id, y_position
                        )
                        panels.append(panel)
                        panel_id += 1
                        y_position += 8
            
            elif dashboard_type == DashboardType.SECURITY:
                # Dashboard sécurité
                security_panels = [
                    {
                        'title': 'Failed Login Attempts',
                        'metrics': ['auth_failed_attempts_total'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Suspicious Activities',
                        'metrics': ['security_events_total'],
                        'type': 'table'
                    },
                    {
                        'title': 'SSL Certificate Status',
                        'metrics': ['ssl_cert_expiry_days'],
                        'type': 'gauge'
                    }
                ]
                
                for panel_config in security_panels:
                    available_metrics = [m for m in panel_config['metrics'] if m in metrics]
                    if available_metrics:
                        panel = await self._create_standard_panel(
                            panel_id, panel_config, available_metrics, tenant_id, y_position
                        )
                        panels.append(panel)
                        panel_id += 1
                        y_position += 8
            
            elif dashboard_type == DashboardType.ML_ANALYTICS:
                # Dashboard ML et Analytics
                ml_panels = [
                    {
                        'title': 'Model Accuracy',
                        'metrics': ['ml_model_accuracy'],
                        'type': 'timeseries'
                    },
                    {
                        'title': 'Prediction Latency',
                        'metrics': ['ml_prediction_duration_seconds'],
                        'type': 'histogram'
                    },
                    {
                        'title': 'Training Jobs',
                        'metrics': ['ml_training_jobs_total'],
                        'type': 'singlestat'
                    },
                    {
                        'title': 'Recommendation Quality',
                        'metrics': ['spotify_recommendation_score'],
                        'type': 'timeseries'
                    }
                ]
                
                for panel_config in ml_panels:
                    available_metrics = [m for m in panel_config['metrics'] if m in metrics]
                    if available_metrics:
                        panel = await self._create_standard_panel(
                            panel_id, panel_config, available_metrics, tenant_id, y_position
                        )
                        panels.append(panel)
                        panel_id += 1
                        y_position += 8
            
            # Ajout de panels génériques pour les métriques restantes
            remaining_metrics = [m for m in metrics if not any(m in p.targets[0]['expr'] for p in panels if p.targets)]
            if remaining_metrics:
                generic_panel = await self._create_generic_metrics_panel(
                    panel_id, remaining_metrics, tenant_id, y_position
                )
                panels.append(generic_panel)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des panels: {e}")
        
        return panels
    
    async def _create_standard_panel(self,
                                   panel_id: int,
                                   config: Dict[str, Any],
                                   metrics: List[str],
                                   tenant_id: str,
                                   y_position: int) -> Panel:
        """Création d'un panel standard"""
        
        # Génération des targets
        targets = []
        for i, metric in enumerate(metrics):
            target = {
                'expr': f'{metric}{{tenant_id="{tenant_id}"}}',
                'refId': chr(65 + i),
                'legendFormat': metric,
                'interval': '30s'
            }
            targets.append(target)
        
        # Position sur la grille
        grid_pos = {
            'h': 8,
            'w': 12,
            'x': (panel_id - 1) % 2 * 12,  # Alternance gauche/droite
            'y': y_position
        }
        
        panel = Panel(
            id=panel_id,
            title=config['title'],
            type=VisualizationType(config.get('type', 'timeseries')),
            datasource=self.default_datasource,
            targets=targets,
            grid_pos=grid_pos
        )
        
        # Configuration spécifique au type
        panel.options = await self._get_panel_options(panel.type, config)
        panel.field_config = await self._get_field_config(panel.type, config)
        
        return panel
    
    async def _get_panel_options(self, panel_type: VisualizationType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration des options d'un panel selon son type"""
        base_options = {}
        
        if panel_type == VisualizationType.TIME_SERIES:
            base_options = {
                'legend': {'displayMode': 'table', 'placement': 'right'},
                'tooltip': {'mode': 'multi', 'sort': 'desc'}
            }
        elif panel_type == VisualizationType.GAUGE:
            base_options = {
                'orientation': 'auto',
                'showThresholdLabels': False,
                'showThresholdMarkers': True
            }
        elif panel_type == VisualizationType.SINGLE_STAT:
            base_options = {
                'colorMode': 'background',
                'graphMode': 'area',
                'justifyMode': 'auto'
            }
        
        # Fusion avec la configuration personnalisée
        custom_options = config.get('options', {})
        base_options.update(custom_options)
        
        return base_options
