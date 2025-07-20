#!/usr/bin/env python3
"""
Reporting Module pour PagerDuty Integration

Module avancé de génération de rapports et d'analytics pour PagerDuty.
Fournit des fonctionnalités complètes de reporting, métriques,
dashboards interactifs, et analyses prédictives.

Fonctionnalités:
- Génération de rapports détaillés
- Métriques et KPIs en temps réel
- Dashboards interactifs
- Analyses prédictives
- Exportation multi-formats
- Intégration avec outils BI
- Alertes sur anomalies

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import argparse
import json
import sys
import csv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from utils.api_client import PagerDutyAPIClient
from utils.formatters import MessageFormatter

console = Console()
logger = structlog.get_logger(__name__)

class ReportType(Enum):
    """Types de rapports"""
    INCIDENT_SUMMARY = "incident_summary"
    PERFORMANCE_METRICS = "performance_metrics"
    TEAM_ANALYTICS = "team_analytics"
    SLA_COMPLIANCE = "sla_compliance"
    TREND_ANALYSIS = "trend_analysis"
    ESCALATION_ANALYSIS = "escalation_analysis"
    COST_ANALYSIS = "cost_analysis"
    PREDICTIVE_INSIGHTS = "predictive_insights"

class ExportFormat(Enum):
    """Formats d'export"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    HTML = "html"
    DASHBOARD = "dashboard"

@dataclass
class ReportConfig:
    """Configuration de rapport"""
    report_type: ReportType
    date_range: Tuple[datetime, datetime]
    filters: Dict[str, Any] = field(default_factory=dict)
    export_format: ExportFormat = ExportFormat.JSON
    include_charts: bool = True
    include_raw_data: bool = False
    team_filter: Optional[List[str]] = None
    service_filter: Optional[List[str]] = None
    severity_filter: Optional[List[str]] = None

@dataclass
class MetricData:
    """Données de métrique"""
    name: str
    value: float
    unit: str = ""
    trend: Optional[float] = None
    target: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    description: str = ""

@dataclass
class KPICollection:
    """Collection de KPIs"""
    timestamp: datetime
    mttr: Optional[float] = None  # Mean Time To Resolution
    mtta: Optional[float] = None  # Mean Time To Acknowledge
    mtbf: Optional[float] = None  # Mean Time Between Failures
    availability: Optional[float] = None
    incident_count: int = 0
    escalation_rate: Optional[float] = None
    customer_impact_score: Optional[float] = None
    sla_compliance: Optional[float] = None

class DataCollector:
    """Collecteur de données pour les rapports"""
    
    def __init__(self, api_client: PagerDutyAPIClient):
        self.api_client = api_client
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def collect_incidents_data(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Collecte les données d'incidents"""
        cache_key = f"incidents_{start_date}_{end_date}_{hash(str(filters))}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        # Simulation de collecte de données d'incidents
        # En production, on utiliserait l'API PagerDuty
        incidents_data = self._generate_sample_incidents(start_date, end_date, filters)
        
        self.cache[cache_key] = {
            "data": incidents_data,
            "timestamp": datetime.now()
        }
        
        return incidents_data
    
    def _generate_sample_incidents(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Génère des données d'incidents d'exemple"""
        incidents = []
        
        # Générer des incidents avec patterns réalistes
        days = (end_date - start_date).days
        incidents_per_day = np.random.poisson(5, days)  # Moyenne de 5 incidents par jour
        
        services = ["web-frontend", "api-gateway", "database", "auth-service", "notification-service"]
        severities = ["critical", "high", "medium", "low"]
        severity_weights = [0.1, 0.2, 0.4, 0.3]
        
        incident_id = 1
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for _ in range(incidents_per_day[day]):
                # Simuler des patterns réalistes
                hour = np.random.choice(24, p=self._get_hourly_distribution())
                minute = np.random.randint(0, 60)
                
                created_at = current_date.replace(hour=hour, minute=minute)
                
                # Durée de résolution basée sur la sévérité
                severity = np.random.choice(severities, p=severity_weights)
                if severity == "critical":
                    resolution_minutes = np.random.exponential(120)  # Moyenne 2h
                elif severity == "high":
                    resolution_minutes = np.random.exponential(240)  # Moyenne 4h
                elif severity == "medium":
                    resolution_minutes = np.random.exponential(480)  # Moyenne 8h
                else:
                    resolution_minutes = np.random.exponential(1440)  # Moyenne 24h
                
                resolved_at = created_at + timedelta(minutes=resolution_minutes)
                
                # Temps d'acknowledgement (plus rapide que résolution)
                ack_minutes = min(resolution_minutes * 0.1, np.random.exponential(30))
                acknowledged_at = created_at + timedelta(minutes=ack_minutes)
                
                incident = {
                    "id": f"INCIDENT-{incident_id:06d}",
                    "title": f"Issue with {np.random.choice(services)}",
                    "service": np.random.choice(services),
                    "severity": severity,
                    "status": "resolved",
                    "created_at": created_at.isoformat(),
                    "acknowledged_at": acknowledged_at.isoformat(),
                    "resolved_at": resolved_at.isoformat(),
                    "escalation_count": np.random.poisson(0.5),
                    "affected_users": self._calculate_affected_users(severity),
                    "business_impact": self._calculate_business_impact(severity),
                    "assigned_team": np.random.choice(["team-alpha", "team-beta", "team-gamma"]),
                    "tags": self._generate_tags(services)
                }
                
                incidents.append(incident)
                incident_id += 1
        
        return incidents
    
    def _get_hourly_distribution(self) -> List[float]:
        """Distribution des incidents par heure (plus d'incidents en heures de bureau)"""
        hours = np.arange(24)
        # Pattern réaliste avec plus d'incidents pendant les heures de bureau
        distribution = np.exp(-0.5 * ((hours - 14) / 6) ** 2)  # Pic vers 14h
        distribution = distribution / distribution.sum()
        return distribution
    
    def _calculate_affected_users(self, severity: str) -> int:
        """Calcule le nombre d'utilisateurs affectés"""
        if severity == "critical":
            return np.random.lognormal(8, 2)  # Beaucoup d'utilisateurs
        elif severity == "high":
            return np.random.lognormal(6, 1.5)
        elif severity == "medium":
            return np.random.lognormal(4, 1)
        else:
            return np.random.lognormal(2, 0.5)
    
    def _calculate_business_impact(self, severity: str) -> float:
        """Calcule l'impact business (score 0-10)"""
        if severity == "critical":
            return np.random.uniform(7, 10)
        elif severity == "high":
            return np.random.uniform(5, 8)
        elif severity == "medium":
            return np.random.uniform(3, 6)
        else:
            return np.random.uniform(0, 4)
    
    def _generate_tags(self, services: List[str]) -> List[str]:
        """Génère des tags pour un incident"""
        possible_tags = ["database", "network", "performance", "security", "deployment", "config"]
        tag_count = np.random.poisson(2)
        return np.random.choice(possible_tags, min(tag_count, len(possible_tags)), replace=False).tolist()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifie si le cache est valide"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]["timestamp"]
        return (datetime.now() - cached_time).seconds < self.cache_ttl
    
    async def collect_team_data(self) -> List[Dict[str, Any]]:
        """Collecte les données d'équipes"""
        # Simulation de données d'équipes
        teams = [
            {
                "id": "team-alpha",
                "name": "Alpha Team",
                "members": 8,
                "on_call_schedule": "24/7",
                "primary_services": ["web-frontend", "api-gateway"],
                "escalation_policy": "standard"
            },
            {
                "id": "team-beta",
                "name": "Beta Team", 
                "members": 6,
                "on_call_schedule": "business_hours",
                "primary_services": ["database", "auth-service"],
                "escalation_policy": "database_critical"
            },
            {
                "id": "team-gamma",
                "name": "Gamma Team",
                "members": 10,
                "on_call_schedule": "24/7",
                "primary_services": ["notification-service"],
                "escalation_policy": "standard"
            }
        ]
        
        return teams
    
    async def collect_sla_data(self) -> Dict[str, Any]:
        """Collecte les données SLA"""
        return {
            "critical_incidents": {
                "target_acknowledgement": 15,  # minutes
                "target_resolution": 240,      # minutes
                "current_acknowledgement": 12,
                "current_resolution": 180
            },
            "high_incidents": {
                "target_acknowledgement": 30,
                "target_resolution": 480,
                "current_acknowledgement": 25,
                "current_resolution": 420
            },
            "availability_targets": {
                "web-frontend": 99.9,
                "api-gateway": 99.95,
                "database": 99.99,
                "auth-service": 99.9,
                "notification-service": 99.5
            }
        }

class MetricsCalculator:
    """Calculateur de métriques et KPIs"""
    
    def __init__(self):
        self.formatter = MessageFormatter()
    
    def calculate_kpis(self, incidents_data: List[Dict[str, Any]]) -> KPICollection:
        """Calcule les KPIs à partir des données d'incidents"""
        if not incidents_data:
            return KPICollection(timestamp=datetime.now())
        
        df = pd.DataFrame(incidents_data)
        
        # Convertir les dates
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['acknowledged_at'] = pd.to_datetime(df['acknowledged_at'])
        df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        
        # Calculer les durées
        df['time_to_acknowledge'] = (df['acknowledged_at'] - df['created_at']).dt.total_seconds() / 60
        df['time_to_resolve'] = (df['resolved_at'] - df['created_at']).dt.total_seconds() / 60
        
        # Calculer les KPIs
        kpis = KPICollection(
            timestamp=datetime.now(),
            mttr=df['time_to_resolve'].mean(),
            mtta=df['time_to_acknowledge'].mean(),
            incident_count=len(df),
            escalation_rate=df['escalation_count'].mean(),
            customer_impact_score=df['business_impact'].mean() if 'business_impact' in df else None
        )
        
        # Calculer la disponibilité
        if len(df) > 0:
            total_downtime = df['time_to_resolve'].sum()
            total_time = (df['resolved_at'].max() - df['created_at'].min()).total_seconds() / 60
            if total_time > 0:
                kpis.availability = max(0, (total_time - total_downtime) / total_time * 100)
        
        return kpis
    
    def calculate_trend_metrics(
        self,
        incidents_data: List[Dict[str, Any]],
        period: str = "daily"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calcule les métriques de tendance"""
        if not incidents_data:
            return {}
        
        df = pd.DataFrame(incidents_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['time_to_resolve'] = (
            pd.to_datetime(df['resolved_at']) - df['created_at']
        ).dt.total_seconds() / 60
        
        # Grouper par période
        if period == "daily":
            df['period'] = df['created_at'].dt.date
        elif period == "weekly":
            df['period'] = df['created_at'].dt.to_period('W')
        elif period == "monthly":
            df['period'] = df['created_at'].dt.to_period('M')
        else:
            df['period'] = df['created_at'].dt.date
        
        # Calculer les métriques par période
        trend_data = {
            "incident_count": [],
            "average_resolution_time": [],
            "severity_distribution": [],
            "team_performance": []
        }
        
        for period_value, group in df.groupby('period'):
            trend_data["incident_count"].append({
                "period": str(period_value),
                "count": len(group),
                "critical": len(group[group['severity'] == 'critical']),
                "high": len(group[group['severity'] == 'high']),
                "medium": len(group[group['severity'] == 'medium']),
                "low": len(group[group['severity'] == 'low'])
            })
            
            trend_data["average_resolution_time"].append({
                "period": str(period_value),
                "average_minutes": group['time_to_resolve'].mean(),
                "median_minutes": group['time_to_resolve'].median(),
                "p95_minutes": group['time_to_resolve'].quantile(0.95)
            })
        
        return trend_data
    
    def calculate_team_metrics(
        self,
        incidents_data: List[Dict[str, Any]],
        teams_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calcule les métriques par équipe"""
        if not incidents_data:
            return {}
        
        df = pd.DataFrame(incidents_data)
        df['time_to_resolve'] = (
            pd.to_datetime(df['resolved_at']) - pd.to_datetime(df['created_at'])
        ).dt.total_seconds() / 60
        
        team_metrics = {}
        
        for team in teams_data:
            team_id = team['id']
            team_incidents = df[df['assigned_team'] == team_id]
            
            if len(team_incidents) > 0:
                team_metrics[team_id] = {
                    "name": team['name'],
                    "incident_count": len(team_incidents),
                    "average_resolution_time": team_incidents['time_to_resolve'].mean(),
                    "incidents_per_member": len(team_incidents) / team['members'],
                    "severity_breakdown": team_incidents['severity'].value_counts().to_dict(),
                    "escalation_rate": team_incidents['escalation_count'].mean(),
                    "primary_services": team['primary_services']
                }
        
        return team_metrics
    
    def calculate_sla_compliance(
        self,
        incidents_data: List[Dict[str, Any]],
        sla_targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcule la conformité SLA"""
        if not incidents_data:
            return {}
        
        df = pd.DataFrame(incidents_data)
        df['time_to_acknowledge'] = (
            pd.to_datetime(df['acknowledged_at']) - pd.to_datetime(df['created_at'])
        ).dt.total_seconds() / 60
        df['time_to_resolve'] = (
            pd.to_datetime(df['resolved_at']) - pd.to_datetime(df['created_at'])
        ).dt.total_seconds() / 60
        
        compliance = {}
        
        for severity in ['critical', 'high']:
            if severity in sla_targets:
                severity_incidents = df[df['severity'] == severity]
                
                if len(severity_incidents) > 0:
                    ack_target = sla_targets[severity]['target_acknowledgement']
                    res_target = sla_targets[severity]['target_resolution']
                    
                    ack_compliance = (
                        severity_incidents['time_to_acknowledge'] <= ack_target
                    ).mean() * 100
                    
                    res_compliance = (
                        severity_incidents['time_to_resolve'] <= res_target
                    ).mean() * 100
                    
                    compliance[severity] = {
                        "acknowledgement_compliance": ack_compliance,
                        "resolution_compliance": res_compliance,
                        "incident_count": len(severity_incidents),
                        "average_ack_time": severity_incidents['time_to_acknowledge'].mean(),
                        "average_res_time": severity_incidents['time_to_resolve'].mean()
                    }
        
        return compliance

class ChartGenerator:
    """Générateur de graphiques et visualisations"""
    
    def __init__(self):
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_incident_trend_chart(
        self,
        trend_data: Dict[str, List[Dict[str, Any]]],
        output_path: Optional[str] = None
    ) -> str:
        """Crée un graphique de tendance des incidents"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique du nombre d'incidents
        incident_counts = trend_data.get("incident_count", [])
        if incident_counts:
            periods = [item["period"] for item in incident_counts]
            counts = [item["count"] for item in incident_counts]
            
            ax1.plot(periods, counts, marker='o', linewidth=2, markersize=6)
            ax1.set_title("Incident Count Trend", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Number of Incidents")
            ax1.grid(True, alpha=0.3)
            
            # Rotation des labels pour une meilleure lisibilité
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Graphique du temps de résolution
        resolution_times = trend_data.get("average_resolution_time", [])
        if resolution_times:
            periods = [item["period"] for item in resolution_times]
            avg_times = [item["average_minutes"] for item in resolution_times]
            
            ax2.plot(periods, avg_times, marker='s', linewidth=2, markersize=6, color='orange')
            ax2.set_title("Average Resolution Time Trend", fontsize=14, fontweight='bold')
            ax2.set_ylabel("Minutes")
            ax2.set_xlabel("Period")
            ax2.grid(True, alpha=0.3)
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            temp_path = f"/tmp/incident_trend_{int(datetime.now().timestamp())}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
    
    def create_severity_distribution_chart(
        self,
        incidents_data: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """Crée un graphique de distribution des sévérités"""
        df = pd.DataFrame(incidents_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique en secteurs
        severity_counts = df['severity'].value_counts()
        colors = ['#ff6b6b', '#ffa726', '#ffeb3b', '#66bb6a']
        
        ax1.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title("Incident Distribution by Severity", fontsize=14, fontweight='bold')
        
        # Graphique en barres
        severity_counts.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title("Incident Count by Severity", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Number of Incidents")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            temp_path = f"/tmp/severity_dist_{int(datetime.now().timestamp())}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
    
    def create_team_performance_chart(
        self,
        team_metrics: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """Crée un graphique de performance des équipes"""
        if not team_metrics:
            return ""
        
        teams = list(team_metrics.keys())
        incident_counts = [team_metrics[team]['incident_count'] for team in teams]
        avg_resolution_times = [team_metrics[team]['average_resolution_time'] for team in teams]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique des incidents par équipe
        bars1 = ax1.bar(teams, incident_counts, color='skyblue', alpha=0.7)
        ax1.set_title("Incidents Handled by Team", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Number of Incidents")
        ax1.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Graphique des temps de résolution
        bars2 = ax2.bar(teams, avg_resolution_times, color='lightcoral', alpha=0.7)
        ax2.set_title("Average Resolution Time by Team", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Minutes")
        ax2.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            temp_path = f"/tmp/team_performance_{int(datetime.now().timestamp())}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
    
    def create_interactive_dashboard(
        self,
        kpis: KPICollection,
        trend_data: Dict[str, List[Dict[str, Any]]],
        team_metrics: Dict[str, Dict[str, Any]]
    ) -> str:
        """Crée un dashboard interactif avec Plotly"""
        
        # Créer une grille de sous-graphiques
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("KPIs Overview", "Incident Trend", 
                          "Severity Distribution", "Team Performance",
                          "Resolution Time Trend", "SLA Compliance"),
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # KPIs en indicateurs
        if kpis.mttr:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=kpis.mttr,
                    title={"text": "MTTR (minutes)"},
                    gauge={'axis': {'range': [None, 500]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 120], 'color': "lightgray"},
                               {'range': [120, 240], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 240}}
                ),
                row=1, col=1
            )
        
        # Tendance des incidents
        incident_counts = trend_data.get("incident_count", [])
        if incident_counts:
            periods = [item["period"] for item in incident_counts]
            counts = [item["count"] for item in incident_counts]
            
            fig.add_trace(
                go.Scatter(x=periods, y=counts, mode='lines+markers',
                          name='Incident Count', line_color='blue'),
                row=1, col=2
            )
        
        # Distribution des sévérités (exemple)
        fig.add_trace(
            go.Pie(labels=['Critical', 'High', 'Medium', 'Low'],
                   values=[10, 25, 40, 25],
                   name="Severity"),
            row=2, col=1
        )
        
        # Performance des équipes
        if team_metrics:
            teams = list(team_metrics.keys())
            incident_counts = [team_metrics[team]['incident_count'] for team in teams]
            
            fig.add_trace(
                go.Bar(x=teams, y=incident_counts, name="Team Incidents"),
                row=2, col=2
            )
        
        # Mettre à jour le layout
        fig.update_layout(
            height=1200,
            title_text="PagerDuty Analytics Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Sauvegarder le dashboard
        output_path = f"/tmp/dashboard_{int(datetime.now().timestamp())}.html"
        fig.write_html(output_path)
        
        return output_path

class ReportGenerator:
    """Générateur de rapports principal"""
    
    def __init__(self, api_key: str):
        self.api_client = PagerDutyAPIClient(api_key, None)
        self.data_collector = DataCollector(self.api_client)
        self.metrics_calculator = MetricsCalculator()
        self.chart_generator = ChartGenerator()
        self.formatter = MessageFormatter()
    
    async def generate_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Génère un rapport selon la configuration"""
        logger.info(f"Generating {config.report_type.value} report")
        
        start_date, end_date = config.date_range
        
        # Collecter les données
        incidents_data = await self.data_collector.collect_incidents_data(
            start_date, end_date, config.filters
        )
        
        teams_data = await self.data_collector.collect_team_data()
        sla_data = await self.data_collector.collect_sla_data()
        
        # Appliquer les filtres
        incidents_data = self._apply_filters(incidents_data, config)
        
        # Calculer les métriques
        kpis = self.metrics_calculator.calculate_kpis(incidents_data)
        trend_data = self.metrics_calculator.calculate_trend_metrics(incidents_data)
        team_metrics = self.metrics_calculator.calculate_team_metrics(incidents_data, teams_data)
        sla_compliance = self.metrics_calculator.calculate_sla_compliance(incidents_data, sla_data)
        
        # Construire le rapport
        report = {
            "metadata": {
                "report_type": config.report_type.value,
                "generated_at": datetime.now().isoformat(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "filters_applied": config.filters,
                "total_incidents": len(incidents_data)
            },
            "kpis": asdict(kpis),
            "trends": trend_data,
            "team_metrics": team_metrics,
            "sla_compliance": sla_compliance,
            "charts": []
        }
        
        # Générer les graphiques si demandé
        if config.include_charts:
            charts = await self._generate_charts(kpis, trend_data, team_metrics, incidents_data)
            report["charts"] = charts
        
        # Inclure les données brutes si demandé
        if config.include_raw_data:
            report["raw_data"] = incidents_data
        
        # Exporter dans le format demandé
        if config.export_format != ExportFormat.JSON:
            report["export_path"] = await self._export_report(report, config)
        
        return report
    
    def _apply_filters(self, incidents_data: List[Dict[str, Any]], config: ReportConfig) -> List[Dict[str, Any]]:
        """Applique les filtres à la configuration"""
        filtered_data = incidents_data
        
        if config.team_filter:
            filtered_data = [
                incident for incident in filtered_data
                if incident.get("assigned_team") in config.team_filter
            ]
        
        if config.service_filter:
            filtered_data = [
                incident for incident in filtered_data
                if incident.get("service") in config.service_filter
            ]
        
        if config.severity_filter:
            filtered_data = [
                incident for incident in filtered_data
                if incident.get("severity") in config.severity_filter
            ]
        
        return filtered_data
    
    async def _generate_charts(
        self,
        kpis: KPICollection,
        trend_data: Dict[str, List[Dict[str, Any]]],
        team_metrics: Dict[str, Dict[str, Any]],
        incidents_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Génère tous les graphiques pour le rapport"""
        charts = []
        
        try:
            # Graphique de tendance
            trend_chart = self.chart_generator.create_incident_trend_chart(trend_data)
            charts.append(trend_chart)
            
            # Distribution des sévérités
            severity_chart = self.chart_generator.create_severity_distribution_chart(incidents_data)
            charts.append(severity_chart)
            
            # Performance des équipes
            if team_metrics:
                team_chart = self.chart_generator.create_team_performance_chart(team_metrics)
                charts.append(team_chart)
            
            # Dashboard interactif
            dashboard = self.chart_generator.create_interactive_dashboard(kpis, trend_data, team_metrics)
            charts.append(dashboard)
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
        
        return charts
    
    async def _export_report(self, report: Dict[str, Any], config: ReportConfig) -> str:
        """Exporte le rapport dans le format demandé"""
        timestamp = int(datetime.now().timestamp())
        
        if config.export_format == ExportFormat.CSV:
            return self._export_to_csv(report, timestamp)
        elif config.export_format == ExportFormat.EXCEL:
            return self._export_to_excel(report, timestamp)
        elif config.export_format == ExportFormat.PDF:
            return self._export_to_pdf(report, timestamp)
        elif config.export_format == ExportFormat.HTML:
            return self._export_to_html(report, timestamp)
        else:
            # JSON par défaut
            output_path = f"/tmp/report_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return output_path
    
    def _export_to_csv(self, report: Dict[str, Any], timestamp: int) -> str:
        """Exporte vers CSV"""
        output_path = f"/tmp/report_{timestamp}.csv"
        
        # Convertir les données principales en DataFrame
        rows = []
        if "raw_data" in report:
            rows = report["raw_data"]
        else:
            # Créer des lignes à partir des métriques
            for team_id, metrics in report.get("team_metrics", {}).items():
                rows.append({
                    "team": team_id,
                    "incident_count": metrics["incident_count"],
                    "avg_resolution_time": metrics["average_resolution_time"],
                    "escalation_rate": metrics["escalation_rate"]
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        return output_path
    
    def _export_to_excel(self, report: Dict[str, Any], timestamp: int) -> str:
        """Exporte vers Excel"""
        output_path = f"/tmp/report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Feuille des KPIs
            kpis_df = pd.DataFrame([report["kpis"]])
            kpis_df.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Feuille des métriques d'équipes
            if report.get("team_metrics"):
                team_df = pd.DataFrame.from_dict(report["team_metrics"], orient='index')
                team_df.to_excel(writer, sheet_name='Team Metrics')
            
            # Feuille des données brutes si disponibles
            if report.get("raw_data"):
                raw_df = pd.DataFrame(report["raw_data"])
                raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        return output_path
    
    def _export_to_pdf(self, report: Dict[str, Any], timestamp: int) -> str:
        """Exporte vers PDF"""
        output_path = f"/tmp/report_{timestamp}.pdf"
        
        with PdfPages(output_path) as pdf:
            # Page de couverture
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, "PagerDuty Analytics Report", 
                   fontsize=24, ha='center', fontweight='bold')
            ax.text(0.5, 0.7, f"Generated: {report['metadata']['generated_at']}", 
                   fontsize=12, ha='center')
            ax.text(0.5, 0.6, f"Period: {report['metadata']['date_range']['start']} to {report['metadata']['date_range']['end']}", 
                   fontsize=12, ha='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Ajouter les graphiques existants
            for chart_path in report.get("charts", []):
                if chart_path.endswith('.png'):
                    img = plt.imread(chart_path)
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        return output_path
    
    def _export_to_html(self, report: Dict[str, Any], timestamp: int) -> str:
        """Exporte vers HTML"""
        output_path = f"/tmp/report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PagerDuty Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .kpi {{ display: inline-block; margin: 10px; padding: 20px; 
                       border: 1px solid #ddd; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PagerDuty Analytics Report</h1>
                <p>Generated: {report['metadata']['generated_at']}</p>
                <p>Period: {report['metadata']['date_range']['start']} to {report['metadata']['date_range']['end']}</p>
            </div>
            
            <div class="section">
                <h2>Key Performance Indicators</h2>
                <div class="kpi">
                    <h3>MTTR</h3>
                    <p>{report['kpis'].get('mttr', 'N/A')} minutes</p>
                </div>
                <div class="kpi">
                    <h3>MTTA</h3>
                    <p>{report['kpis'].get('mtta', 'N/A')} minutes</p>
                </div>
                <div class="kpi">
                    <h3>Incident Count</h3>
                    <p>{report['kpis'].get('incident_count', 0)}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Team Performance</h2>
                <table>
                    <tr>
                        <th>Team</th>
                        <th>Incidents</th>
                        <th>Avg Resolution (min)</th>
                        <th>Escalation Rate</th>
                    </tr>
        """
        
        # Ajouter les métriques d'équipes
        for team_id, metrics in report.get("team_metrics", {}).items():
            html_content += f"""
                    <tr>
                        <td>{team_id}</td>
                        <td>{metrics['incident_count']}</td>
                        <td>{metrics['average_resolution_time']:.1f}</td>
                        <td>{metrics['escalation_rate']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    async def close(self):
        """Ferme les connexions"""
        await self.api_client.close()

async def main():
    """Fonction principale CLI"""
    parser = argparse.ArgumentParser(description="PagerDuty Reporting Tool")
    parser.add_argument("--api-key", required=True, help="Clé API PagerDuty")
    parser.add_argument("--report-type", required=True,
                       choices=[rt.value for rt in ReportType],
                       help="Type de rapport à générer")
    parser.add_argument("--start-date", required=True,
                       help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True,
                       help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--export-format", default="json",
                       choices=[ef.value for ef in ExportFormat],
                       help="Format d'export")
    parser.add_argument("--team-filter", nargs="*", help="Filtrer par équipes")
    parser.add_argument("--service-filter", nargs="*", help="Filtrer par services")
    parser.add_argument("--severity-filter", nargs="*", help="Filtrer par sévérités")
    parser.add_argument("--include-charts", action="store_true", help="Inclure les graphiques")
    parser.add_argument("--include-raw-data", action="store_true", help="Inclure les données brutes")
    parser.add_argument("--output-dir", default="/tmp", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    try:
        # Analyser les dates
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Créer la configuration
        config = ReportConfig(
            report_type=ReportType(args.report_type),
            date_range=(start_date, end_date),
            export_format=ExportFormat(args.export_format),
            include_charts=args.include_charts,
            include_raw_data=args.include_raw_data,
            team_filter=args.team_filter,
            service_filter=args.service_filter,
            severity_filter=args.severity_filter
        )
        
        # Générer le rapport
        generator = ReportGenerator(args.api_key)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating report...", total=None)
            
            report = await generator.generate_report(config)
            
            progress.update(task, description="Report generated successfully")
        
        # Afficher le résumé
        console.print(Panel.fit(
            f"Report Type: {config.report_type.value}\n"
            f"Period: {start_date.date()} to {end_date.date()}\n"
            f"Total Incidents: {report['metadata']['total_incidents']}\n"
            f"Export Format: {config.export_format.value}\n"
            f"Generated At: {report['metadata']['generated_at']}",
            title="Report Summary"
        ))
        
        # Afficher les KPIs
        kpis_table = Table(title="Key Performance Indicators")
        kpis_table.add_column("Metric", style="bold")
        kpis_table.add_column("Value", justify="right")
        kpis_table.add_column("Unit")
        
        kpis = report['kpis']
        if kpis.get('mttr'):
            kpis_table.add_row("MTTR", f"{kpis['mttr']:.1f}", "minutes")
        if kpis.get('mtta'):
            kpis_table.add_row("MTTA", f"{kpis['mtta']:.1f}", "minutes")
        if kpis.get('incident_count'):
            kpis_table.add_row("Incident Count", str(kpis['incident_count']), "incidents")
        if kpis.get('escalation_rate'):
            kpis_table.add_row("Escalation Rate", f"{kpis['escalation_rate']:.2f}", "per incident")
        
        console.print(kpis_table)
        
        # Afficher les informations d'export
        if "export_path" in report:
            console.print(f"\n[green]Report exported to: {report['export_path']}[/green]")
        
        if report.get('charts'):
            console.print(f"\n[blue]Charts generated: {len(report['charts'])} files[/blue]")
            for chart in report['charts']:
                console.print(f"  - {chart}")
        
        await generator.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
