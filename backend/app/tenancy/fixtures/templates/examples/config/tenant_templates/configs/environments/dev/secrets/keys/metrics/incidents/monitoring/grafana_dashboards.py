# =============================================================================
# Monitoring & Observability - Grafana Dashboards Enterprise
# =============================================================================
# 
# Générateur de dashboards Grafana enterprise avec visualisations avancées,
# métriques ML et tableaux de bord interactifs.
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture dashboards et visualisations)
# - DevOps Senior Engineer (Configuration Grafana et déploiement)
# - Data Visualization Specialist (Design UX/UI des dashboards)
# - Monitoring Expert (Métriques et alerting avancé)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import yaml
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES DE DASHBOARDS
# =============================================================================

class VisualizationType(Enum):
    """Types de visualisations Grafana"""
    GRAPH = "graph"
    SINGLE_STAT = "singlestat"
    TABLE = "table"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BAR_GAUGE = "bargauge"
    PIE_CHART = "piechart"
    STAT = "stat"
    GAUGE = "gauge"
    LOGS = "logs"
    ALERT_LIST = "alertlist"
    DASHBOARD_LIST = "dashlist"
    TEXT = "text"
    NEWS = "news"

class TimeRange(Enum):
    """Plages de temps prédéfinies"""
    LAST_5M = "now-5m"
    LAST_15M = "now-15m"
    LAST_30M = "now-30m"
    LAST_1H = "now-1h"
    LAST_3H = "now-3h"
    LAST_6H = "now-6h"
    LAST_12H = "now-12h"
    LAST_24H = "now-24h"
    LAST_7D = "now-7d"
    LAST_30D = "now-30d"

class GridPosition:
    """Position dans la grille Grafana"""
    def __init__(self, x: int = 0, y: int = 0, w: int = 12, h: int = 8):
        self.x = x
        self.y = y
        self.w = w  # largeur
        self.h = h  # hauteur

@dataclass
class GrafanaTarget:
    """Requête pour un panel Grafana"""
    expr: str  # Requête PromQL
    legend_format: str = ""
    ref_id: str = "A"
    interval: str = ""
    format: str = "time_series"
    instant: bool = False

@dataclass
class GrafanaPanel:
    """Panel Grafana"""
    id: int
    title: str
    type: VisualizationType
    targets: List[GrafanaTarget]
    grid_pos: GridPosition
    datasource: str = "Prometheus"
    description: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    alert: Optional[Dict[str, Any]] = None

@dataclass
class GrafanaDashboard:
    """Dashboard Grafana complet"""
    id: Optional[int]
    title: str
    description: str
    tags: List[str]
    panels: List[GrafanaPanel]
    time_range: TimeRange = TimeRange.LAST_1H
    refresh: str = "30s"
    uid: str = ""
    version: int = 1
    editable: bool = True
    style: str = "dark"
    timezone: str = "browser"
    variables: List[Dict[str, Any]] = field(default_factory=list)

# =============================================================================
# GÉNÉRATEUR DE DASHBOARDS ENTERPRISE
# =============================================================================

class GrafanaDashboardGenerator:
    """
    Générateur de dashboards Grafana enterprise avec templates avancés,
    métriques ML et personnalisation par tenant.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates_dir = Path(config.get('templates_dir', './grafana_templates'))
        self.output_dir = Path(config.get('output_dir', './grafana_dashboards'))
        self.datasource_name = config.get('datasource_name', 'Prometheus')
        
        # Création des répertoires
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter pour IDs de panels
        self.panel_id_counter = 1
        
        logger.info("GrafanaDashboardGenerator initialisé")

    async def initialize(self):
        """Initialisation du générateur"""
        try:
            # Création des templates par défaut
            await self.create_default_templates()
            
            logger.info("Générateur de dashboards initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation générateur: {e}")
            raise

    async def create_default_templates(self):
        """Création des templates de dashboards par défaut"""
        
        # Dashboard système overview
        await self.create_system_overview_dashboard()
        
        # Dashboard incidents
        await self.create_incidents_dashboard()
        
        # Dashboard API performance
        await self.create_api_performance_dashboard()
        
        # Dashboard ML monitoring
        await self.create_ml_monitoring_dashboard()
        
        # Dashboard business metrics
        await self.create_business_metrics_dashboard()
        
        # Dashboard alerting overview
        await self.create_alerting_dashboard()

    async def create_system_overview_dashboard(self):
        """Dashboard d'overview système"""
        
        panels = []
        
        # Panel CPU Usage
        cpu_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="CPU Usage",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                    legend_format="CPU Usage %",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 12, 8),
            datasource=self.datasource_name,
            description="System CPU usage percentage",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 70},
                            {"color": "red", "value": 85}
                        ]
                    }
                }
            }
        )
        panels.append(cpu_panel)
        
        # Panel Memory Usage
        memory_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Memory Usage",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="(1 - (system_memory_usage_bytes{type=\"available\"} / system_memory_usage_bytes{type=\"total\"})) * 100",
                    legend_format="Memory Usage %",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(12, 0, 12, 8),
            datasource=self.datasource_name,
            description="System memory usage percentage",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 80},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        )
        panels.append(memory_panel)
        
        # Panel Disk Usage
        disk_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Disk Usage",
            type=VisualizationType.BAR_GAUGE,
            targets=[
                GrafanaTarget(
                    expr="(system_disk_usage_bytes{type=\"used\"} / system_disk_usage_bytes{type=\"total\"}) * 100",
                    legend_format="{{device}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 24, 8),
            datasource=self.datasource_name,
            description="Disk usage per device",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 80},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        )
        panels.append(disk_panel)
        
        # Panel Network Traffic
        network_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Network Traffic",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="rate(system_network_bytes_total{direction=\"sent\"}[5m]) * 8",
                    legend_format="{{interface}} sent",
                    ref_id="A"
                ),
                GrafanaTarget(
                    expr="rate(system_network_bytes_total{direction=\"received\"}[5m]) * 8",
                    legend_format="{{interface}} received",
                    ref_id="B"
                )
            ],
            grid_pos=GridPosition(0, 16, 24, 8),
            datasource=self.datasource_name,
            description="Network traffic in bits per second",
            field_config={
                "defaults": {
                    "unit": "bps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "barAlignment": 0,
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"},
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "axisColorMode": "text",
                        "scaleDistribution": {"type": "linear"},
                        "axisCenteredZero": False,
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "thresholdsStyle": {"mode": "off"}
                    }
                }
            }
        )
        panels.append(network_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="System Overview",
            description="Comprehensive system monitoring dashboard",
            tags=["system", "infrastructure", "monitoring"],
            panels=panels,
            time_range=TimeRange.LAST_1H,
            refresh="30s",
            uid="system-overview",
            variables=[
                {
                    "name": "instance",
                    "type": "query",
                    "query": "label_values(up, instance)",
                    "label": "Instance",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                }
            ]
        )
        
        await self.save_dashboard(dashboard, "system_overview.json")

    async def create_incidents_dashboard(self):
        """Dashboard de monitoring des incidents"""
        
        panels = []
        
        # Panel Incidents Total
        incidents_total_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Total Incidents",
            type=VisualizationType.STAT,
            targets=[
                GrafanaTarget(
                    expr="sum(incidents_total)",
                    legend_format="Total Incidents",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 6, 8),
            datasource=self.datasource_name,
            description="Total number of incidents",
            field_config={
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False}
                    },
                    "mappings": [],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        )
        panels.append(incidents_total_panel)
        
        # Panel Incidents Rate
        incidents_rate_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Incident Rate",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="rate(incidents_total[5m])",
                    legend_format="{{severity}} - {{tenant_id}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(6, 0, 18, 8),
            datasource=self.datasource_name,
            description="Incident creation rate per minute",
            field_config={
                "defaults": {
                    "unit": "rps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "normal", "group": "A"}
                    }
                }
            }
        )
        panels.append(incidents_rate_panel)
        
        # Panel Active Incidents
        active_incidents_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Active Incidents by Severity",
            type=VisualizationType.PIE_CHART,
            targets=[
                GrafanaTarget(
                    expr="sum by (severity) (incidents_active)",
                    legend_format="{{severity}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 12, 8),
            datasource=self.datasource_name,
            description="Current active incidents grouped by severity",
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "pieType": "pie",
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {
                    "displayMode": "list",
                    "placement": "right",
                    "showLegend": True
                },
                "displayLabels": ["name", "value"]
            }
        )
        panels.append(active_incidents_panel)
        
        # Panel Incident Duration
        incident_duration_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Incident Resolution Time",
            type=VisualizationType.HISTOGRAM,
            targets=[
                GrafanaTarget(
                    expr="histogram_quantile(0.95, incidents_duration_seconds)",
                    legend_format="95th percentile",
                    ref_id="A"
                ),
                GrafanaTarget(
                    expr="histogram_quantile(0.50, incidents_duration_seconds)",
                    legend_format="50th percentile",
                    ref_id="B"
                )
            ],
            grid_pos=GridPosition(12, 8, 12, 8),
            datasource=self.datasource_name,
            description="Incident resolution time distribution",
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False}
                    }
                }
            }
        )
        panels.append(incident_duration_panel)
        
        # Panel Incidents by Category
        incidents_category_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Incidents by Category",
            type=VisualizationType.TABLE,
            targets=[
                GrafanaTarget(
                    expr="sum by (category, severity) (increase(incidents_total[1h]))",
                    legend_format="",
                    ref_id="A",
                    format="table"
                )
            ],
            grid_pos=GridPosition(0, 16, 24, 8),
            datasource=self.datasource_name,
            description="Incidents breakdown by category and severity in the last hour",
            field_config={
                "defaults": {
                    "custom": {
                        "align": "auto",
                        "cellOptions": {"type": "auto"},
                        "inspect": False
                    },
                    "mappings": [],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        )
        panels.append(incidents_category_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="Incidents Monitoring",
            description="Comprehensive incidents monitoring and analytics",
            tags=["incidents", "monitoring", "analytics"],
            panels=panels,
            time_range=TimeRange.LAST_6H,
            refresh="1m",
            uid="incidents-monitoring",
            variables=[
                {
                    "name": "tenant",
                    "type": "query",
                    "query": "label_values(incidents_total, tenant_id)",
                    "label": "Tenant",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                },
                {
                    "name": "severity",
                    "type": "query",
                    "query": "label_values(incidents_total, severity)",
                    "label": "Severity",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                }
            ]
        )
        
        await self.save_dashboard(dashboard, "incidents_monitoring.json")

    async def create_api_performance_dashboard(self):
        """Dashboard de performance API"""
        
        panels = []
        
        # Panel Request Rate
        request_rate_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="API Request Rate",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="sum(rate(api_requests_total[5m])) by (method, endpoint)",
                    legend_format="{{method}} {{endpoint}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 12, 8),
            datasource=self.datasource_name,
            description="API request rate per endpoint",
            field_config={
                "defaults": {
                    "unit": "rps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"}
                    }
                }
            }
        )
        panels.append(request_rate_panel)
        
        # Panel Response Time
        response_time_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="API Response Time",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
                    legend_format="95th percentile",
                    ref_id="A"
                ),
                GrafanaTarget(
                    expr="histogram_quantile(0.50, rate(api_request_duration_seconds_bucket[5m]))",
                    legend_format="50th percentile",
                    ref_id="B"
                ),
                GrafanaTarget(
                    expr="histogram_quantile(0.99, rate(api_request_duration_seconds_bucket[5m]))",
                    legend_format="99th percentile",
                    ref_id="C"
                )
            ],
            grid_pos=GridPosition(12, 0, 12, 8),
            datasource=self.datasource_name,
            description="API response time percentiles",
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5
                    }
                }
            }
        )
        panels.append(response_time_panel)
        
        # Panel Error Rate
        error_rate_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="API Error Rate",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="sum(rate(api_requests_total{status_code=~\"4..|5..\"}[5m])) / sum(rate(api_requests_total[5m])) * 100",
                    legend_format="Error Rate %",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 24, 8),
            datasource=self.datasource_name,
            description="API error rate percentage",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 5}
                        ]
                    },
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 2,
                        "fillOpacity": 20,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5
                    }
                }
            }
        )
        panels.append(error_rate_panel)
        
        # Panel Top Endpoints
        top_endpoints_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Top API Endpoints",
            type=VisualizationType.TABLE,
            targets=[
                GrafanaTarget(
                    expr="topk(10, sum by (endpoint, method) (rate(api_requests_total[5m])))",
                    legend_format="",
                    ref_id="A",
                    format="table"
                )
            ],
            grid_pos=GridPosition(0, 16, 24, 8),
            datasource=self.datasource_name,
            description="Top 10 API endpoints by request rate",
            field_config={
                "defaults": {
                    "custom": {
                        "align": "auto",
                        "cellOptions": {"type": "auto"},
                        "inspect": False
                    },
                    "mappings": [],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        )
        panels.append(top_endpoints_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="API Performance",
            description="API performance monitoring and analytics",
            tags=["api", "performance", "monitoring"],
            panels=panels,
            time_range=TimeRange.LAST_1H,
            refresh="30s",
            uid="api-performance",
            variables=[
                {
                    "name": "endpoint",
                    "type": "query",
                    "query": "label_values(api_requests_total, endpoint)",
                    "label": "Endpoint",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                }
            ]
        )
        
        await self.save_dashboard(dashboard, "api_performance.json")

    async def create_ml_monitoring_dashboard(self):
        """Dashboard de monitoring ML"""
        
        panels = []
        
        # Panel ML Predictions
        ml_predictions_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="ML Model Predictions",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="sum(rate(ml_model_predictions_total[5m])) by (model_name, result)",
                    legend_format="{{model_name}} - {{result}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 12, 8),
            datasource=self.datasource_name,
            description="ML model prediction rate by result",
            field_config={
                "defaults": {
                    "unit": "rps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "normal", "group": "A"}
                    }
                }
            }
        )
        panels.append(ml_predictions_panel)
        
        # Panel ML Inference Time
        ml_inference_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="ML Inference Time",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m]))",
                    legend_format="95th percentile",
                    ref_id="A"
                ),
                GrafanaTarget(
                    expr="histogram_quantile(0.50, rate(ml_model_inference_duration_seconds_bucket[5m]))",
                    legend_format="50th percentile",
                    ref_id="B"
                )
            ],
            grid_pos=GridPosition(12, 0, 12, 8),
            datasource=self.datasource_name,
            description="ML model inference time percentiles",
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5
                    }
                }
            }
        )
        panels.append(ml_inference_panel)
        
        # Panel Model Accuracy
        model_accuracy_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Model Accuracy",
            type=VisualizationType.GAUGE,
            targets=[
                GrafanaTarget(
                    expr="ml_model_accuracy",
                    legend_format="{{model_name}} v{{model_version}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 12, 8),
            datasource=self.datasource_name,
            description="Current model accuracy scores",
            field_config={
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.7},
                            {"color": "green", "value": 0.9}
                        ]
                    }
                }
            },
            options={
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
        )
        panels.append(model_accuracy_panel)
        
        # Panel Anomaly Detection
        anomaly_detection_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Anomaly Detection",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="rate(anomalies_detected_total[5m])",
                    legend_format="{{detector_type}} - {{severity}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(12, 8, 12, 8),
            datasource=self.datasource_name,
            description="Anomaly detection rate by type and severity",
            field_config={
                "defaults": {
                    "unit": "rps",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "normal", "group": "A"}
                    }
                }
            }
        )
        panels.append(anomaly_detection_panel)
        
        # Panel Anomaly Scores
        anomaly_scores_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Anomaly Scores Distribution",
            type=VisualizationType.HEATMAP,
            targets=[
                GrafanaTarget(
                    expr="increase(anomaly_score_bucket[5m])",
                    legend_format="{{le}}",
                    ref_id="A",
                    format="heatmap"
                )
            ],
            grid_pos=GridPosition(0, 16, 24, 8),
            datasource=self.datasource_name,
            description="Heatmap of anomaly scores distribution",
            field_config={
                "defaults": {
                    "custom": {
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "scaleDistribution": {"type": "linear"}
                    }
                }
            }
        )
        panels.append(anomaly_scores_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="ML Monitoring",
            description="Machine Learning models monitoring and performance",
            tags=["ml", "ai", "monitoring", "anomaly"],
            panels=panels,
            time_range=TimeRange.LAST_1H,
            refresh="1m",
            uid="ml-monitoring",
            variables=[
                {
                    "name": "model",
                    "type": "query",
                    "query": "label_values(ml_model_predictions_total, model_name)",
                    "label": "Model",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                }
            ]
        )
        
        await self.save_dashboard(dashboard, "ml_monitoring.json")

    async def create_business_metrics_dashboard(self):
        """Dashboard de métriques business"""
        
        panels = []
        
        # Panel Active Users
        active_users_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Active Users",
            type=VisualizationType.STAT,
            targets=[
                GrafanaTarget(
                    expr="sum(users_active{time_range=\"24h\"})",
                    legend_format="24h Active Users",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 6, 8),
            datasource=self.datasource_name,
            description="Active users in the last 24 hours",
            field_config={
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        )
        panels.append(active_users_panel)
        
        # Panel Cost Overview
        cost_overview_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Cost Overview",
            type=VisualizationType.PIE_CHART,
            targets=[
                GrafanaTarget(
                    expr="sum by (cost_type) (cost_total_usd)",
                    legend_format="{{cost_type}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(6, 0, 12, 8),
            datasource=self.datasource_name,
            description="Cost breakdown by type",
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "pieType": "pie",
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {
                    "displayMode": "list",
                    "placement": "right",
                    "showLegend": True
                },
                "displayLabels": ["name", "value"]
            }
        )
        panels.append(cost_overview_panel)
        
        # Panel SLA Uptime
        sla_uptime_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="SLA Uptime",
            type=VisualizationType.GAUGE,
            targets=[
                GrafanaTarget(
                    expr="avg(sla_uptime_percent)",
                    legend_format="Average Uptime",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(18, 0, 6, 8),
            datasource=self.datasource_name,
            description="Average SLA uptime percentage",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 95,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 99},
                            {"color": "green", "value": 99.9}
                        ]
                    }
                }
            },
            options={
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
        )
        panels.append(sla_uptime_panel)
        
        # Panel Automation Success Rate
        automation_success_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Automation Success Rate",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr="sum(rate(automation_executions_total{status=\"success\"}[5m])) / sum(rate(automation_executions_total[5m])) * 100",
                    legend_format="Success Rate %",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 24, 8),
            datasource=self.datasource_name,
            description="Automation execution success rate",
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 90},
                            {"color": "green", "value": 95}
                        ]
                    },
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 2,
                        "fillOpacity": 20,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5
                    }
                }
            }
        )
        panels.append(automation_success_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="Business Metrics",
            description="Business KPIs and operational metrics",
            tags=["business", "kpi", "operations"],
            panels=panels,
            time_range=TimeRange.LAST_24H,
            refresh="5m",
            uid="business-metrics",
            variables=[
                {
                    "name": "tenant",
                    "type": "query",
                    "query": "label_values(users_active, tenant_id)",
                    "label": "Tenant",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True
                }
            ]
        )
        
        await self.save_dashboard(dashboard, "business_metrics.json")

    async def create_alerting_dashboard(self):
        """Dashboard d'overview des alertes"""
        
        panels = []
        
        # Panel Active Alerts
        active_alerts_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title="Active Alerts",
            type=VisualizationType.ALERT_LIST,
            targets=[],
            grid_pos=GridPosition(0, 0, 24, 16),
            datasource=self.datasource_name,
            description="List of currently active alerts",
            options={
                "showOptions": "current",
                "maxItems": 20,
                "sortOrder": 1,
                "dashboardAlerts": False,
                "alertName": "",
                "dashboardTitle": "",
                "folderId": None,
                "tags": []
            }
        )
        panels.append(active_alerts_panel)
        
        # Création du dashboard
        dashboard = GrafanaDashboard(
            id=None,
            title="Alerting Overview",
            description="Overview of all system alerts and their status",
            tags=["alerts", "monitoring", "overview"],
            panels=panels,
            time_range=TimeRange.LAST_1H,
            refresh="30s",
            uid="alerting-overview"
        )
        
        await self.save_dashboard(dashboard, "alerting_overview.json")

    async def save_dashboard(self, dashboard: GrafanaDashboard, filename: str):
        """Sauvegarde d'un dashboard au format JSON"""
        try:
            dashboard_json = self.dashboard_to_json(dashboard)
            
            output_file = self.output_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dashboard sauvegardé: {output_file}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde dashboard {filename}: {e}")

    def dashboard_to_json(self, dashboard: GrafanaDashboard) -> Dict[str, Any]:
        """Conversion d'un dashboard en JSON Grafana"""
        
        panels_json = []
        for panel in dashboard.panels:
            panel_json = {
                "id": panel.id,
                "title": panel.title,
                "type": panel.type.value,
                "datasource": {
                    "type": "prometheus",
                    "uid": panel.datasource
                },
                "gridPos": {
                    "h": panel.grid_pos.h,
                    "w": panel.grid_pos.w,
                    "x": panel.grid_pos.x,
                    "y": panel.grid_pos.y
                },
                "targets": [
                    {
                        "expr": target.expr,
                        "legendFormat": target.legend_format,
                        "refId": target.ref_id,
                        "interval": target.interval,
                        "format": target.format,
                        "instant": target.instant
                    }
                    for target in panel.targets
                ],
                "fieldConfig": panel.field_config,
                "options": panel.options
            }
            
            if panel.description:
                panel_json["description"] = panel.description
            
            if panel.alert:
                panel_json["alert"] = panel.alert
            
            panels_json.append(panel_json)
        
        dashboard_json = {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "tags": dashboard.tags,
            "style": dashboard.style,
            "timezone": dashboard.timezone,
            "editable": dashboard.editable,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "links": [],
            "liveNow": False,
            "panels": panels_json,
            "refresh": dashboard.refresh,
            "schemaVersion": 38,
            "time": {
                "from": dashboard.time_range.value,
                "to": "now"
            },
            "timepicker": {},
            "templating": {
                "list": dashboard.variables
            },
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {
                            "type": "grafana",
                            "uid": "-- Grafana --"
                        },
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "uid": dashboard.uid,
            "version": dashboard.version,
            "weekStart": ""
        }
        
        return dashboard_json

    def get_next_panel_id(self) -> int:
        """Récupération du prochain ID de panel"""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id

    async def generate_tenant_dashboards(self, tenant_id: str):
        """Génération de dashboards personnalisés pour un tenant"""
        try:
            # Dashboard spécifique au tenant basé sur les templates
            tenant_dashboard = await self.create_tenant_specific_dashboard(tenant_id)
            await self.save_dashboard(tenant_dashboard, f"tenant_{tenant_id}_overview.json")
            
            logger.info(f"Dashboard tenant généré: {tenant_id}")
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard tenant {tenant_id}: {e}")

    async def create_tenant_specific_dashboard(self, tenant_id: str) -> GrafanaDashboard:
        """Création d'un dashboard spécifique à un tenant"""
        
        panels = []
        
        # Panel incidents du tenant
        tenant_incidents_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title=f"Incidents - Tenant {tenant_id}",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr=f"sum(rate(incidents_total{{tenant_id=\"{tenant_id}\"}}[5m])) by (severity)",
                    legend_format="{{severity}}",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 0, 24, 8),
            datasource=self.datasource_name,
            description=f"Incident rate for tenant {tenant_id}"
        )
        panels.append(tenant_incidents_panel)
        
        # Panel API performance du tenant
        tenant_api_panel = GrafanaPanel(
            id=self.get_next_panel_id(),
            title=f"API Performance - Tenant {tenant_id}",
            type=VisualizationType.GRAPH,
            targets=[
                GrafanaTarget(
                    expr=f"histogram_quantile(0.95, rate(api_request_duration_seconds_bucket{{tenant_id=\"{tenant_id}\"}}[5m]))",
                    legend_format="95th percentile",
                    ref_id="A"
                )
            ],
            grid_pos=GridPosition(0, 8, 24, 8),
            datasource=self.datasource_name,
            description=f"API response time for tenant {tenant_id}"
        )
        panels.append(tenant_api_panel)
        
        dashboard = GrafanaDashboard(
            id=None,
            title=f"Tenant {tenant_id} Overview",
            description=f"Dedicated monitoring dashboard for tenant {tenant_id}",
            tags=["tenant", tenant_id, "monitoring"],
            panels=panels,
            time_range=TimeRange.LAST_1H,
            refresh="1m",
            uid=f"tenant-{tenant_id}-overview"
        )
        
        return dashboard

# =============================================================================
# GESTIONNAIRE D'EXPORT GRAFANA
# =============================================================================

class GrafanaExporter:
    """Gestionnaire d'export et de déploiement des dashboards Grafana"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grafana_url = config.get('grafana_url', 'http://localhost:3000')
        self.api_key = config.get('api_key', '')
        self.org_id = config.get('org_id', 1)
        
        logger.info("GrafanaExporter initialisé")

    async def deploy_dashboard(self, dashboard_json: Dict[str, Any]) -> bool:
        """Déploiement d'un dashboard vers Grafana"""
        try:
            # Simulation du déploiement (remplacer par une vraie API call)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # URL de l'API Grafana
            url = f"{self.grafana_url}/api/dashboards/db"
            
            # Payload pour l'API
            payload = {
                "dashboard": dashboard_json,
                "overwrite": True,
                "message": f"Deployed via automation at {datetime.utcnow()}"
            }
            
            # Simulation de la requête
            logger.info(f"Déploiement dashboard: {dashboard_json.get('title', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur déploiement dashboard: {e}")
            return False

    async def deploy_all_dashboards(self, dashboards_dir: Path) -> int:
        """Déploiement de tous les dashboards d'un répertoire"""
        deployed_count = 0
        
        try:
            for dashboard_file in dashboards_dir.glob("*.json"):
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    dashboard_json = json.load(f)
                
                if await self.deploy_dashboard(dashboard_json):
                    deployed_count += 1
                    logger.info(f"Dashboard déployé: {dashboard_file.name}")
                else:
                    logger.error(f"Échec déploiement: {dashboard_file.name}")
            
            logger.info(f"Déploiement terminé: {deployed_count} dashboards")
            
        except Exception as e:
            logger.error(f"Erreur déploiement massif: {e}")
        
        return deployed_count

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du générateur de dashboards"""
    
    config = {
        'templates_dir': './grafana_templates',
        'output_dir': './grafana_dashboards',
        'datasource_name': 'Prometheus',
        'grafana_url': 'http://localhost:3000',
        'api_key': 'your_grafana_api_key',
        'org_id': 1
    }
    
    # Initialisation du générateur
    generator = GrafanaDashboardGenerator(config)
    await generator.initialize()
    
    # Génération de dashboards pour des tenants spécifiques
    tenants = ["tenant_123", "tenant_456", "tenant_789"]
    for tenant_id in tenants:
        await generator.generate_tenant_dashboards(tenant_id)
    
    # Export vers Grafana
    exporter = GrafanaExporter(config)
    deployed_count = await exporter.deploy_all_dashboards(generator.output_dir)
    
    print(f"=== Génération et déploiement terminés ===")
    print(f"Dashboards générés: {len(list(generator.output_dir.glob('*.json')))}")
    print(f"Dashboards déployés: {deployed_count}")

if __name__ == "__main__":
    asyncio.run(main())
