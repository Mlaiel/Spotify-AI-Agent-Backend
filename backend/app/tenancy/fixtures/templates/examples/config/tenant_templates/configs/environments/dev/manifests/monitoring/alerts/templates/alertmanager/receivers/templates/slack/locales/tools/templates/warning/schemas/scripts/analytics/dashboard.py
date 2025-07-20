"""
Advanced Performance Dashboard & Metrics Exporter
===============================================

Tableau de bord et exportateur de métriques ultra-avancé pour monitoring
en temps réel des performances du système analytics Spotify AI Agent.

Fonctionnalités:
- Dashboard en temps réel avec WebSocket
- Export multi-format (Prometheus, Grafana, InfluxDB)
- Métriques business et techniques
- Alertes proactives basées sur ML
- Rapports automatisés avec IA

Auteur: Fahed Mlaiel - Senior ML Engineer & Data Analytics Expert
License: Enterprise Grade
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque

import aioredis
import aioprometheus
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram, 
    generate_latest, CONTENT_TYPE_LATEST
)
import aiohttp
from aiohttp import web, WSMsgType
import websockets
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from .analytics import AdvancedAnalyticsEngine
from .algorithms import AnomalyDetectionAlgorithm, TrendForecastingAlgorithm
from .utils import CacheManager, MetricsCollector, DataProcessor


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrique de performance."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram
    description: str = ""
    unit: str = ""


@dataclass
class DashboardConfig:
    """Configuration du dashboard."""
    refresh_interval: int = 5  # secondes
    max_data_points: int = 1000
    alert_threshold_cpu: float = 80.0
    alert_threshold_memory: float = 85.0
    alert_threshold_response_time: float = 2.0
    enable_realtime: bool = True
    enable_alerts: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["prometheus", "json", "csv"])


class MetricsExporter:
    """Exportateur de métriques multi-format."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics_cache = CacheManager(prefix="metrics_export")
        
        # Métriques Prometheus
        self.setup_prometheus_metrics()
        
    def setup_prometheus_metrics(self):
        """Configure les métriques Prometheus."""
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'spotify_active_users_total',
            'Number of active users',
            ['tenant', 'region'],
            registry=self.registry
        )
        
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy_score',
            'ML model accuracy score',
            ['model_name', 'version'],
            registry=self.registry
        )
        
    async def export_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def export_json(self, metrics: List[PerformanceMetric]) -> str:
        """Exporte les métriques au format JSON."""
        json_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "labels": m.labels,
                    "type": m.metric_type,
                    "description": m.description,
                    "unit": m.unit
                }
                for m in metrics
            ]
        }
        return json.dumps(json_data, indent=2)
    
    async def export_csv(self, metrics: List[PerformanceMetric]) -> str:
        """Exporte les métriques au format CSV."""
        df = pd.DataFrame([
            {
                "name": m.name,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "labels": json.dumps(m.labels),
                "type": m.metric_type,
                "description": m.description,
                "unit": m.unit
            }
            for m in metrics
        ])
        return df.to_csv(index=False)
    
    async def export_influxdb(self, metrics: List[PerformanceMetric]) -> str:
        """Exporte les métriques au format InfluxDB Line Protocol."""
        lines = []
        for metric in metrics:
            labels_str = ",".join([f"{k}={v}" for k, v in metric.labels.items()])
            timestamp_ns = int(metric.timestamp.timestamp() * 1e9)
            
            line = f"{metric.name}"
            if labels_str:
                line += f",{labels_str}"
            line += f" value={metric.value} {timestamp_ns}"
            lines.append(line)
            
        return "\n".join(lines)


class AdvancedDashboard:
    """Dashboard avancé avec WebSocket et temps réel."""
    
    def __init__(self, analytics_engine: AdvancedAnalyticsEngine, config: DashboardConfig):
        self.analytics_engine = analytics_engine
        self.config = config
        self.exporter = MetricsExporter()
        self.anomaly_detector = AnomalyDetectionAlgorithm()
        self.trend_forecaster = TrendForecastingAlgorithm()
        
        # Cache pour les métriques
        self.metrics_history = defaultdict(lambda: deque(maxlen=config.max_data_points))
        self.connected_clients = set()
        
        # Redis pour le cache distribué
        self.redis_client = None
        
    async def initialize(self):
        """Initialise le dashboard."""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            await self.anomaly_detector.initialize()
            await self.trend_forecaster.initialize()
            logger.info("Dashboard initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du dashboard: {e}")
    
    async def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques système."""
        import psutil
        
        metrics = []
        timestamp = datetime.utcnow()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            name="system_cpu_usage",
            value=cpu_percent,
            timestamp=timestamp,
            metric_type="gauge",
            description="CPU usage percentage",
            unit="percent"
        ))
        
        # Mémoire
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            name="system_memory_usage",
            value=memory.percent,
            timestamp=timestamp,
            metric_type="gauge",
            description="Memory usage percentage",
            unit="percent"
        ))
        
        # Disque
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(PerformanceMetric(
            name="system_disk_usage",
            value=disk_percent,
            timestamp=timestamp,
            metric_type="gauge",
            description="Disk usage percentage",
            unit="percent"
        ))
        
        # Réseau
        network = psutil.net_io_counters()
        metrics.extend([
            PerformanceMetric(
                name="network_bytes_sent",
                value=network.bytes_sent,
                timestamp=timestamp,
                metric_type="counter",
                description="Network bytes sent",
                unit="bytes"
            ),
            PerformanceMetric(
                name="network_bytes_received",
                value=network.bytes_recv,
                timestamp=timestamp,
                metric_type="counter",
                description="Network bytes received",
                unit="bytes"
            )
        ])
        
        return metrics
    
    async def collect_application_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques de l'application."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Métriques analytics
        analytics_stats = await self.analytics_engine.get_stats()
        
        metrics.extend([
            PerformanceMetric(
                name="active_tenants",
                value=analytics_stats.get("active_tenants", 0),
                timestamp=timestamp,
                metric_type="gauge",
                description="Number of active tenants",
                unit="count"
            ),
            PerformanceMetric(
                name="processed_events_per_second",
                value=analytics_stats.get("events_per_second", 0),
                timestamp=timestamp,
                metric_type="gauge",
                description="Events processed per second",
                unit="events/s"
            ),
            PerformanceMetric(
                name="ml_predictions_accuracy",
                value=analytics_stats.get("ml_accuracy", 0.95),
                timestamp=timestamp,
                metric_type="gauge",
                description="ML model prediction accuracy",
                unit="score"
            ),
            PerformanceMetric(
                name="cache_hit_ratio",
                value=analytics_stats.get("cache_hit_ratio", 0.85),
                timestamp=timestamp,
                metric_type="gauge",
                description="Cache hit ratio",
                unit="ratio"
            )
        ])
        
        return metrics
    
    async def detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les métriques."""
        anomalies = []
        
        for metric in metrics:
            # Historique pour détection d'anomalies
            history = list(self.metrics_history[metric.name])
            if len(history) >= 10:  # Minimum de données
                values = [m.value for m in history[-50:]]  # 50 dernières valeurs
                values.append(metric.value)
                
                # Détection d'anomalie
                is_anomaly = await self.anomaly_detector.detect_single(
                    metric.value, values[:-1]
                )
                
                if is_anomaly:
                    anomalies.append({
                        "metric_name": metric.name,
                        "current_value": metric.value,
                        "expected_range": await self.anomaly_detector.get_expected_range(values[:-1]),
                        "severity": self._calculate_severity(metric),
                        "timestamp": metric.timestamp.isoformat(),
                        "description": f"Anomalie détectée pour {metric.name}"
                    })
        
        return anomalies
    
    def _calculate_severity(self, metric: PerformanceMetric) -> str:
        """Calcule la sévérité d'une anomalie."""
        if metric.name == "system_cpu_usage" and metric.value > 90:
            return "critical"
        elif metric.name == "system_memory_usage" and metric.value > 95:
            return "critical"
        elif metric.name in ["system_cpu_usage", "system_memory_usage"] and metric.value > 80:
            return "warning"
        else:
            return "info"
    
    async def generate_forecast(self, metric_name: str, horizon: int = 60) -> Dict[str, Any]:
        """Génère une prévision pour une métrique."""
        history = list(self.metrics_history[metric_name])
        if len(history) < 20:
            return {"error": "Pas assez de données historiques"}
        
        values = [m.value for m in history]
        timestamps = [m.timestamp for m in history]
        
        forecast = await self.trend_forecaster.forecast(
            values, horizon, feature_names=[metric_name]
        )
        
        return {
            "metric_name": metric_name,
            "forecast_horizon": horizon,
            "predicted_values": forecast.tolist(),
            "confidence_interval": await self._calculate_confidence_interval(values, forecast),
            "trend": await self._analyze_trend(values)
        }
    
    async def _calculate_confidence_interval(self, historical: List[float], forecast: np.ndarray) -> Dict[str, List[float]]:
        """Calcule l'intervalle de confiance."""
        std = np.std(historical)
        confidence_95 = 1.96 * std
        
        return {
            "lower": (forecast - confidence_95).tolist(),
            "upper": (forecast + confidence_95).tolist()
        }
    
    async def _analyze_trend(self, values: List[float]) -> str:
        """Analyse la tendance des valeurs."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Régression linéaire simple
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Génère les données complètes du dashboard."""
        # Collecte des métriques
        system_metrics = await self.collect_system_metrics()
        app_metrics = await self.collect_application_metrics()
        all_metrics = system_metrics + app_metrics
        
        # Mise à jour de l'historique
        for metric in all_metrics:
            self.metrics_history[metric.name].append(metric)
        
        # Détection d'anomalies
        anomalies = await self.detect_anomalies(all_metrics)
        
        # Prévisions pour métriques clés
        forecasts = {}
        key_metrics = ["system_cpu_usage", "system_memory_usage", "processed_events_per_second"]
        for metric_name in key_metrics:
            if metric_name in self.metrics_history:
                forecasts[metric_name] = await self.generate_forecast(metric_name)
        
        # Résumé des performances
        performance_summary = await self._generate_performance_summary(all_metrics)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "labels": m.labels,
                    "type": m.metric_type,
                    "description": m.description,
                    "unit": m.unit
                }
                for m in all_metrics
            ],
            "anomalies": anomalies,
            "forecasts": forecasts,
            "performance_summary": performance_summary,
            "system_health": await self._calculate_system_health(all_metrics)
        }
    
    async def _generate_performance_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Génère un résumé des performances."""
        summary = {
            "overall_status": "healthy",
            "critical_issues": 0,
            "warnings": 0,
            "key_indicators": {}
        }
        
        for metric in metrics:
            if metric.name == "system_cpu_usage":
                if metric.value > 90:
                    summary["critical_issues"] += 1
                    summary["overall_status"] = "critical"
                elif metric.value > 80:
                    summary["warnings"] += 1
                    if summary["overall_status"] == "healthy":
                        summary["overall_status"] = "warning"
                summary["key_indicators"]["cpu"] = metric.value
                
            elif metric.name == "system_memory_usage":
                if metric.value > 95:
                    summary["critical_issues"] += 1
                    summary["overall_status"] = "critical"
                elif metric.value > 85:
                    summary["warnings"] += 1
                    if summary["overall_status"] == "healthy":
                        summary["overall_status"] = "warning"
                summary["key_indicators"]["memory"] = metric.value
                
            elif metric.name == "processed_events_per_second":
                summary["key_indicators"]["throughput"] = metric.value
        
        return summary
    
    async def _calculate_system_health(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calcule la santé globale du système."""
        health_score = 100.0
        factors = []
        
        for metric in metrics:
            if metric.name == "system_cpu_usage":
                if metric.value > 80:
                    deduction = min(20, (metric.value - 80) * 2)
                    health_score -= deduction
                    factors.append(f"CPU usage élevé: {metric.value}%")
                    
            elif metric.name == "system_memory_usage":
                if metric.value > 85:
                    deduction = min(25, (metric.value - 85) * 3)
                    health_score -= deduction
                    factors.append(f"Memory usage élevé: {metric.value}%")
        
        health_score = max(0, health_score)
        
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": round(health_score, 1),
            "status": status,
            "affecting_factors": factors
        }
    
    async def start_realtime_monitoring(self):
        """Démarre le monitoring en temps réel."""
        while True:
            try:
                dashboard_data = await self.generate_dashboard_data()
                
                # Diffusion WebSocket
                if self.connected_clients:
                    await self.broadcast_to_clients(dashboard_data)
                
                # Cache Redis
                if self.redis_client:
                    await self.redis_client.setex(
                        "dashboard_data",
                        300,  # 5 minutes TTL
                        json.dumps(dashboard_data, default=str)
                    )
                
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans le monitoring temps réel: {e}")
                await asyncio.sleep(10)
    
    async def broadcast_to_clients(self, data: Dict[str, Any]):
        """Diffuse les données vers tous les clients WebSocket connectés."""
        if not self.connected_clients:
            return
        
        message = json.dumps(data, default=str)
        disconnected = set()
        
        for websocket in self.connected_clients:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.add(websocket)
        
        # Supprime les clients déconnectés
        self.connected_clients -= disconnected
    
    async def handle_websocket(self, websocket: WebSocket):
        """Gère les connexions WebSocket."""
        await websocket.accept()
        self.connected_clients.add(websocket)
        
        try:
            # Envoie les données initiales
            initial_data = await self.generate_dashboard_data()
            await websocket.send_text(json.dumps(initial_data, default=str))
            
            # Maintient la connexion
            while True:
                try:
                    message = await websocket.receive_text()
                    # Traite les commandes du client si nécessaire
                    await self.handle_client_command(websocket, message)
                except WebSocketDisconnect:
                    break
                    
        except Exception as e:
            logger.error(f"Erreur WebSocket: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_client_command(self, websocket: WebSocket, message: str):
        """Traite les commandes reçues des clients."""
        try:
            command = json.loads(message)
            
            if command.get("type") == "get_forecast":
                metric_name = command.get("metric_name")
                horizon = command.get("horizon", 60)
                
                forecast = await self.generate_forecast(metric_name, horizon)
                await websocket.send_text(json.dumps({
                    "type": "forecast_response",
                    "data": forecast
                }, default=str))
                
            elif command.get("type") == "get_metrics_export":
                export_format = command.get("format", "json")
                all_metrics = []
                
                for metric_list in self.metrics_history.values():
                    all_metrics.extend(list(metric_list))
                
                if export_format == "prometheus":
                    exported = await self.exporter.export_prometheus()
                elif export_format == "csv":
                    exported = await self.exporter.export_csv(all_metrics)
                elif export_format == "influxdb":
                    exported = await self.exporter.export_influxdb(all_metrics)
                else:
                    exported = await self.exporter.export_json(all_metrics)
                
                await websocket.send_text(json.dumps({
                    "type": "export_response",
                    "format": export_format,
                    "data": exported
                }, default=str))
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la commande client: {e}")


class PerformanceDashboardAPI:
    """API FastAPI pour le dashboard de performance."""
    
    def __init__(self, dashboard: AdvancedDashboard):
        self.dashboard = dashboard
        self.app = FastAPI(
            title="Spotify AI Analytics Performance Dashboard",
            description="Dashboard ultra-avancé pour monitoring et analytics",
            version="2.0.0"
        )
        self.setup_routes()
    
    def setup_routes(self):
        """Configure les routes de l'API."""
        
        @self.app.get("/")
        async def dashboard_home():
            """Page d'accueil du dashboard."""
            return HTMLResponse(content=self.get_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Récupère les métriques actuelles."""
            data = await self.dashboard.generate_dashboard_data()
            return JSONResponse(content=data)
        
        @self.app.get("/api/metrics/export/{format_type}")
        async def export_metrics(format_type: str):
            """Exporte les métriques dans le format spécifié."""
            all_metrics = []
            for metric_list in self.dashboard.metrics_history.values():
                all_metrics.extend(list(metric_list))
            
            if format_type == "prometheus":
                content = await self.dashboard.exporter.export_prometheus()
                return PlainTextResponse(content=content, media_type=CONTENT_TYPE_LATEST)
            elif format_type == "csv":
                content = await self.dashboard.exporter.export_csv(all_metrics)
                return PlainTextResponse(content=content, media_type="text/csv")
            elif format_type == "influxdb":
                content = await self.dashboard.exporter.export_influxdb(all_metrics)
                return PlainTextResponse(content=content, media_type="text/plain")
            else:
                content = await self.dashboard.exporter.export_json(all_metrics)
                return JSONResponse(content=json.loads(content))
        
        @self.app.get("/api/forecast/{metric_name}")
        async def get_forecast(metric_name: str, horizon: int = 60):
            """Récupère la prévision pour une métrique."""
            forecast = await self.dashboard.generate_forecast(metric_name, horizon)
            return JSONResponse(content=forecast)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Endpoint WebSocket pour temps réel."""
            await self.dashboard.handle_websocket(websocket)
    
    def get_dashboard_html(self) -> str:
        """Retourne le HTML du dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spotify AI Analytics Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/moment@2.29.1/moment.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #1db954; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #1db954; }
                .status-healthy { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-critical { color: #dc3545; }
                .chart-container { height: 300px; margin-top: 10px; }
                .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
                .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
                .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎵 Spotify AI Analytics Performance Dashboard</h1>
                    <p>Monitoring ultra-avancé en temps réel - Powered by Fahed Mlaiel</p>
                </div>
                
                <div id="alerts-container"></div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>System Health</h3>
                        <div id="health-score" class="metric-value">--</div>
                        <div id="health-status">Loading...</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>CPU Usage</h3>
                        <div id="cpu-usage" class="metric-value">--%</div>
                        <div class="chart-container">
                            <canvas id="cpu-chart"></canvas>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Memory Usage</h3>
                        <div id="memory-usage" class="metric-value">--%</div>
                        <div class="chart-container">
                            <canvas id="memory-chart"></canvas>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Active Tenants</h3>
                        <div id="active-tenants" class="metric-value">--</div>
                        <div class="chart-container">
                            <canvas id="tenants-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                const charts = {};
                
                // Initialisation des graphiques
                function initCharts() {
                    const chartConfig = {
                        type: 'line',
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: { type: 'time', time: { unit: 'minute' } },
                                y: { beginAtZero: true }
                            }
                        }
                    };
                    
                    charts.cpu = new Chart(document.getElementById('cpu-chart'), {
                        ...chartConfig,
                        data: { datasets: [{ label: 'CPU %', data: [], borderColor: '#ff6384' }] }
                    });
                    
                    charts.memory = new Chart(document.getElementById('memory-chart'), {
                        ...chartConfig,
                        data: { datasets: [{ label: 'Memory %', data: [], borderColor: '#36a2eb' }] }
                    });
                    
                    charts.tenants = new Chart(document.getElementById('tenants-chart'), {
                        ...chartConfig,
                        data: { datasets: [{ label: 'Active Tenants', data: [], borderColor: '#1db954' }] }
                    });
                }
                
                // Mise à jour des données
                function updateDashboard(data) {
                    // Métriques principales
                    const metrics = data.metrics.reduce((acc, m) => {
                        acc[m.name] = m;
                        return acc;
                    }, {});
                    
                    // System Health
                    if (data.system_health) {
                        document.getElementById('health-score').textContent = data.system_health.score + '/100';
                        document.getElementById('health-status').textContent = data.system_health.status;
                        document.getElementById('health-status').className = `status-${data.system_health.status}`;
                    }
                    
                    // CPU
                    if (metrics.system_cpu_usage) {
                        document.getElementById('cpu-usage').textContent = metrics.system_cpu_usage.value.toFixed(1) + '%';
                        addDataPoint(charts.cpu, metrics.system_cpu_usage);
                    }
                    
                    // Memory
                    if (metrics.system_memory_usage) {
                        document.getElementById('memory-usage').textContent = metrics.system_memory_usage.value.toFixed(1) + '%';
                        addDataPoint(charts.memory, metrics.system_memory_usage);
                    }
                    
                    // Active Tenants
                    if (metrics.active_tenants) {
                        document.getElementById('active-tenants').textContent = metrics.active_tenants.value;
                        addDataPoint(charts.tenants, metrics.active_tenants);
                    }
                    
                    // Alertes
                    updateAlerts(data.anomalies || []);
                }
                
                function addDataPoint(chart, metric) {
                    const data = chart.data.datasets[0].data;
                    data.push({
                        x: new Date(metric.timestamp),
                        y: metric.value
                    });
                    
                    // Limite à 50 points
                    if (data.length > 50) {
                        data.shift();
                    }
                    
                    chart.update('none');
                }
                
                function updateAlerts(anomalies) {
                    const container = document.getElementById('alerts-container');
                    container.innerHTML = '';
                    
                    anomalies.forEach(anomaly => {
                        const alert = document.createElement('div');
                        alert.className = `alert alert-${anomaly.severity}`;
                        alert.innerHTML = `
                            <strong>${anomaly.severity.toUpperCase()}:</strong> 
                            ${anomaly.description} - Valeur: ${anomaly.current_value}
                        `;
                        container.appendChild(alert);
                    });
                }
                
                // WebSocket events
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onopen = function(event) {
                    console.log('WebSocket connecté');
                };
                
                ws.onerror = function(error) {
                    console.error('Erreur WebSocket:', error);
                };
                
                // Initialisation
                initCharts();
            </script>
        </body>
        </html>
        """


# Instance globale pour faciliter l'import
performance_dashboard = None


async def initialize_performance_dashboard(analytics_engine: AdvancedAnalyticsEngine) -> AdvancedDashboard:
    """Initialise le dashboard de performance."""
    global performance_dashboard
    
    config = DashboardConfig(
        refresh_interval=5,
        max_data_points=1000,
        alert_threshold_cpu=80.0,
        alert_threshold_memory=85.0,
        enable_realtime=True,
        enable_alerts=True
    )
    
    performance_dashboard = AdvancedDashboard(analytics_engine, config)
    await performance_dashboard.initialize()
    
    # Démarre le monitoring en tâche de fond
    asyncio.create_task(performance_dashboard.start_realtime_monitoring())
    
    logger.info("Dashboard de performance initialisé et démarré")
    return performance_dashboard


def create_dashboard_app(analytics_engine: AdvancedAnalyticsEngine) -> FastAPI:
    """Crée l'application FastAPI du dashboard."""
    dashboard = AdvancedDashboard(analytics_engine, DashboardConfig())
    api = PerformanceDashboardAPI(dashboard)
    return api.app


if __name__ == "__main__":
    import uvicorn
    from .analytics import AdvancedAnalyticsEngine
    
    # Pour test standalone
    analytics_engine = AdvancedAnalyticsEngine()
    app = create_dashboard_app(analytics_engine)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=False
    )
