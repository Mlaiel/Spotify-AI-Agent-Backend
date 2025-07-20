"""
Scripts d'Analytics Avancés pour Monitoring
Système d'analyse intelligente et reporting pour Spotify AI Agent

Fonctionnalités:
- Analytics temps réel des métriques de performance
- Détection de patterns et anomalies par ML
- Rapports automatisés et dashboards dynamiques
- Prédictions de tendances et capacité
- Analyse de corrélation multi-métriques
- Recommandations d'optimisation basées sur l'IA
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import redis.asyncio as redis

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types d'analyses disponibles"""
    PERFORMANCE_TRENDS = "performance_trends"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPACITY_PLANNING = "capacity_planning"
    CORRELATION_ANALYSIS = "correlation_analysis"
    BUSINESS_METRICS = "business_metrics"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    COST_OPTIMIZATION = "cost_optimization"
    USER_BEHAVIOR = "user_behavior"

class ReportFormat(Enum):
    """Formats de rapport disponibles"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"

@dataclass
class MetricDataPoint:
    """Point de données métrique"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    tenant_id: Optional[str] = None

@dataclass
class AnalyticsReport:
    """Rapport d'analyse"""
    report_id: str
    report_type: AnalyticsType
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    visualizations: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class AnalyticsConfig:
    """Configuration d'analyse"""
    config_id: str
    name: str
    analytics_type: AnalyticsType
    metrics: List[str]
    time_window: timedelta
    update_frequency: timedelta
    enabled: bool = True
    tenant_id: Optional[str] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)

class AdvancedAnalyticsEngine:
    """Moteur d'analytics avancé avec IA"""
    
    def __init__(self):
        self.redis_client = None
        self.metric_cache = {}
        self.analysis_cache = {}
        self.configs: List[AnalyticsConfig] = []
        self.reports_history: List[AnalyticsReport] = []
        
        self._initialize_default_configs()

    async def initialize(self):
        """Initialise le moteur d'analytics"""
        try:
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                db=2
            )
            logger.info("Moteur d'analytics initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")

    def _initialize_default_configs(self):
        """Initialise les configurations d'analyse par défaut"""
        
        # Configuration analyse des tendances de performance
        performance_config = AnalyticsConfig(
            config_id="performance_trends_analysis",
            name="Analyse des tendances de performance",
            analytics_type=AnalyticsType.PERFORMANCE_TRENDS,
            metrics=["cpu_usage", "memory_usage", "response_time", "throughput"],
            time_window=timedelta(hours=24),
            update_frequency=timedelta(minutes=15),
            thresholds={"cpu_usage": 80, "memory_usage": 75, "response_time": 1000}
        )
        
        # Configuration détection d'anomalies
        anomaly_config = AnalyticsConfig(
            config_id="anomaly_detection_analysis",
            name="Détection d'anomalies ML",
            analytics_type=AnalyticsType.ANOMALY_DETECTION,
            metrics=["error_rate", "latency", "audio_quality_score"],
            time_window=timedelta(hours=6),
            update_frequency=timedelta(minutes=5)
        )
        
        # Configuration planification de capacité
        capacity_config = AnalyticsConfig(
            config_id="capacity_planning_analysis",
            name="Planification de capacité",
            analytics_type=AnalyticsType.CAPACITY_PLANNING,
            metrics=["cpu_usage", "memory_usage", "disk_usage", "network_bandwidth"],
            time_window=timedelta(days=7),
            update_frequency=timedelta(hours=6)
        )
        
        self.configs.extend([performance_config, anomaly_config, capacity_config])

    async def collect_metrics_data(self, metrics: List[str], time_range: Tuple[datetime, datetime], tenant_id: Optional[str] = None) -> List[MetricDataPoint]:
        """Collecte les données de métriques sur une période"""
        
        try:
            data_points = []
            start_time, end_time = time_range
            
            # Simulation de collecte de données (en production, intégrer avec Prometheus/InfluxDB)
            current_time = start_time
            while current_time <= end_time:
                for metric in metrics:
                    value = await self._generate_sample_metric_value(metric, current_time)
                    
                    data_point = MetricDataPoint(
                        timestamp=current_time,
                        metric_name=metric,
                        value=value,
                        tenant_id=tenant_id
                    )
                    data_points.append(data_point)
                
                current_time += timedelta(minutes=5)
            
            logger.info(f"Collecté {len(data_points)} points de données pour {len(metrics)} métriques")
            return data_points
            
        except Exception as e:
            logger.error(f"Erreur lors de la collecte de métriques: {e}")
            return []

    async def _generate_sample_metric_value(self, metric_name: str, timestamp: datetime) -> float:
        """Génère des valeurs de métriques simulées réalistes"""
        
        # Patterns de base avec variations temporelles
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Facteur de charge basé sur l'heure (simulation heures de pointe)
        load_factor = 1.0
        if 8 <= hour <= 22:  # Heures de bureau
            load_factor = 1.5
        if 19 <= hour <= 21:  # Pic du soir
            load_factor = 2.0
        
        # Facteur week-end
        if day_of_week >= 5:  # Week-end
            load_factor *= 0.7
        
        # Valeurs de base par métrique
        base_values = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "response_time": 200.0,
            "throughput": 1000.0,
            "error_rate": 1.5,
            "disk_usage": 55.0,
            "network_bandwidth": 800.0,
            "audio_quality_score": 92.0,
            "latency": 50.0
        }
        
        base_value = base_values.get(metric_name, 50.0)
        
        # Application du facteur de charge et bruit aléatoire
        noise = np.random.normal(0, base_value * 0.1)
        value = base_value * load_factor + noise
        
        # Contraintes spécifiques par métrique
        if metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            value = max(0, min(100, value))
        elif metric_name == "error_rate":
            value = max(0, min(20, value))
        elif metric_name == "audio_quality_score":
            value = max(50, min(100, value))
        else:
            value = max(0, value)
        
        return round(value, 2)

    async def generate_performance_trends_report(self, config: AnalyticsConfig) -> AnalyticsReport:
        """Génère un rapport d'analyse des tendances de performance"""
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - config.time_window
            
            # Collecte des données
            data_points = await self.collect_metrics_data(
                config.metrics, 
                (start_time, end_time), 
                config.tenant_id
            )
            
            if not data_points:
                raise Exception("Aucune donnée collectée")
            
            # Conversion en DataFrame pour analyse
            df = pd.DataFrame([
                {
                    'timestamp': dp.timestamp,
                    'metric': dp.metric_name,
                    'value': dp.value
                }
                for dp in data_points
            ])
            
            # Analyse des tendances
            trends = {}
            insights = []
            recommendations = []
            
            for metric in config.metrics:
                metric_data = df[df['metric'] == metric].copy()
                metric_data = metric_data.sort_values('timestamp')
                
                # Calcul de la tendance (régression linéaire simple)
                if len(metric_data) > 1:
                    x = np.arange(len(metric_data))
                    y = metric_data['value'].values
                    trend_slope = np.polyfit(x, y, 1)[0]
                    
                    # Statistiques
                    current_value = y[-1]
                    avg_value = np.mean(y)
                    max_value = np.max(y)
                    min_value = np.min(y)
                    
                    trends[metric] = {
                        'current_value': current_value,
                        'average_value': avg_value,
                        'max_value': max_value,
                        'min_value': min_value,
                        'trend_slope': trend_slope,
                        'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                        'volatility': np.std(y)
                    }
                    
                    # Insights basés sur les seuils
                    threshold = config.thresholds.get(metric)
                    if threshold and current_value > threshold:
                        insights.append(f"⚠️ {metric} dépasse le seuil ({current_value:.1f} > {threshold})")
                        recommendations.append(f"Optimiser {metric} - valeur actuelle critique")
                    
                    # Insights sur les tendances
                    if abs(trend_slope) > avg_value * 0.01:  # Tendance significative
                        direction = "augmentation" if trend_slope > 0 else "diminution"
                        insights.append(f"📈 {metric}: {direction} notable détectée")
                        
                        if trend_slope > 0 and metric in ["cpu_usage", "memory_usage", "error_rate"]:
                            recommendations.append(f"Surveiller {metric} - tendance à la hausse préoccupante")
            
            # Génération des visualisations
            visualizations = await self._generate_performance_visualizations(df, trends)
            
            report = AnalyticsReport(
                report_id=f"perf_trends_{int(datetime.utcnow().timestamp())}",
                report_type=AnalyticsType.PERFORMANCE_TRENDS,
                generated_at=datetime.utcnow(),
                time_range=(start_time, end_time),
                data=trends,
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
                tenant_id=config.tenant_id,
                confidence_score=0.85
            )
            
            self.reports_history.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport tendances: {e}")
            raise

    async def generate_anomaly_detection_report(self, config: AnalyticsConfig) -> AnalyticsReport:
        """Génère un rapport de détection d'anomalies"""
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - config.time_window
            
            data_points = await self.collect_metrics_data(
                config.metrics,
                (start_time, end_time),
                config.tenant_id
            )
            
            # Conversion en DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': dp.timestamp,
                    'metric': dp.metric_name,
                    'value': dp.value
                }
                for dp in data_points
            ])
            
            anomalies = {}
            insights = []
            recommendations = []
            
            for metric in config.metrics:
                metric_data = df[df['metric'] == metric]['value'].values
                
                if len(metric_data) > 10:
                    # Détection d'anomalies par IQR
                    Q1 = np.percentile(metric_data, 25)
                    Q3 = np.percentile(metric_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Identification des anomalies
                    anomaly_indices = np.where(
                        (metric_data < lower_bound) | (metric_data > upper_bound)
                    )[0]
                    
                    anomaly_count = len(anomaly_indices)
                    anomaly_percentage = (anomaly_count / len(metric_data)) * 100
                    
                    anomalies[metric] = {
                        'total_points': len(metric_data),
                        'anomaly_count': anomaly_count,
                        'anomaly_percentage': anomaly_percentage,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'recent_anomalies': anomaly_indices[-5:].tolist() if anomaly_count > 0 else []
                    }
                    
                    # Insights
                    if anomaly_percentage > 5:
                        insights.append(f"🚨 {metric}: {anomaly_percentage:.1f}% d'anomalies détectées")
                        recommendations.append(f"Investiguer les causes des anomalies de {metric}")
                    elif anomaly_percentage > 2:
                        insights.append(f"⚠️ {metric}: {anomaly_percentage:.1f}% d'anomalies (surveillance recommandée)")
            
            # Détection d'anomalies multi-variées
            multivariate_anomalies = await self._detect_multivariate_anomalies(df)
            if multivariate_anomalies['anomaly_count'] > 0:
                insights.append(f"🔍 {multivariate_anomalies['anomaly_count']} anomalies multi-variées détectées")
                recommendations.append("Analyser les corrélations entre métriques pour anomalies complexes")
            
            visualizations = await self._generate_anomaly_visualizations(df, anomalies)
            
            report = AnalyticsReport(
                report_id=f"anomaly_detection_{int(datetime.utcnow().timestamp())}",
                report_type=AnalyticsType.ANOMALY_DETECTION,
                generated_at=datetime.utcnow(),
                time_range=(start_time, end_time),
                data={**anomalies, 'multivariate': multivariate_anomalies},
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
                tenant_id=config.tenant_id,
                confidence_score=0.92
            )
            
            self.reports_history.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport anomalies: {e}")
            raise

    async def generate_capacity_planning_report(self, config: AnalyticsConfig) -> AnalyticsReport:
        """Génère un rapport de planification de capacité"""
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - config.time_window
            
            data_points = await self.collect_metrics_data(
                config.metrics,
                (start_time, end_time),
                config.tenant_id
            )
            
            df = pd.DataFrame([
                {
                    'timestamp': dp.timestamp,
                    'metric': dp.metric_name,
                    'value': dp.value
                }
                for dp in data_points
            ])
            
            capacity_analysis = {}
            insights = []
            recommendations = []
            
            for metric in config.metrics:
                metric_data = df[df['metric'] == metric].copy()
                metric_data = metric_data.sort_values('timestamp')
                
                if len(metric_data) > 20:
                    values = metric_data['value'].values
                    
                    # Prédiction de tendance (régression polynomiale)
                    x = np.arange(len(values))
                    
                    # Prédiction pour les 30 prochains jours
                    future_points = 30 * 24 * 12  # 30 jours en points de 5 min
                    x_future = np.arange(len(values), len(values) + future_points)
                    
                    # Modèle de prédiction simple (en production, utiliser des modèles plus sophistiqués)
                    if len(values) > 10:
                        poly_coeffs = np.polyfit(x, values, 2)
                        future_values = np.polyval(poly_coeffs, x_future)
                        
                        current_value = values[-1]
                        predicted_30d = future_values[-1] if len(future_values) > 0 else current_value
                        growth_rate = ((predicted_30d - current_value) / current_value) * 100
                        
                        # Calcul du temps avant saturation (pour métriques de capacité)
                        saturation_threshold = 90 if metric.endswith('_usage') else None
                        time_to_saturation = None
                        
                        if saturation_threshold and growth_rate > 0:
                            remaining_capacity = saturation_threshold - current_value
                            if remaining_capacity > 0:
                                time_to_saturation = remaining_capacity / (growth_rate / 30)  # jours
                        
                        capacity_analysis[metric] = {
                            'current_value': current_value,
                            'predicted_30d': predicted_30d,
                            'growth_rate_monthly': growth_rate,
                            'time_to_saturation_days': time_to_saturation,
                            'trend_confidence': 0.75,
                            'peak_usage': np.max(values),
                            'average_usage': np.mean(values)
                        }
                        
                        # Insights et recommandations
                        if time_to_saturation and time_to_saturation < 30:
                            insights.append(f"🚨 {metric}: Saturation prévue dans {time_to_saturation:.0f} jours")
                            recommendations.append(f"Planifier extension de capacité pour {metric}")
                        elif time_to_saturation and time_to_saturation < 90:
                            insights.append(f"⚠️ {metric}: Saturation prévue dans {time_to_saturation:.0f} jours")
                            recommendations.append(f"Surveiller {metric} et préparer montée en charge")
                        
                        if growth_rate > 20:
                            insights.append(f"📈 {metric}: Croissance rapide de {growth_rate:.1f}% par mois")
                            recommendations.append(f"Analyser les causes de croissance de {metric}")
            
            # Analyse de corrélation des capacités
            correlation_analysis = await self._analyze_capacity_correlations(df)
            
            visualizations = await self._generate_capacity_visualizations(df, capacity_analysis)
            
            report = AnalyticsReport(
                report_id=f"capacity_planning_{int(datetime.utcnow().timestamp())}",
                report_type=AnalyticsType.CAPACITY_PLANNING,
                generated_at=datetime.utcnow(),
                time_range=(start_time, end_time),
                data={**capacity_analysis, 'correlations': correlation_analysis},
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
                tenant_id=config.tenant_id,
                confidence_score=0.88
            )
            
            self.reports_history.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport capacité: {e}")
            raise

    async def _detect_multivariate_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Détecte les anomalies multi-variées"""
        
        try:
            # Pivot pour avoir les métriques en colonnes
            pivot_df = df.pivot(index='timestamp', columns='metric', values='value')
            pivot_df = pivot_df.fillna(method='forward').fillna(method='backward')
            
            if len(pivot_df) < 10:
                return {'anomaly_count': 0, 'method': 'insufficient_data'}
            
            # Normalisation
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pivot_df.values)
            
            # PCA pour réduction dimensionnelle
            pca = PCA(n_components=min(2, scaled_data.shape[1]))
            pca_data = pca.fit_transform(scaled_data)
            
            # Clustering pour détecter les outliers
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(pca_data)
            
            # Points éloignés du centre des clusters = anomalies
            distances = []
            for i, point in enumerate(pca_data):
                cluster_center = kmeans.cluster_centers_[clusters[i]]
                distance = np.linalg.norm(point - cluster_center)
                distances.append(distance)
            
            # Seuil d'anomalie (95e percentile)
            anomaly_threshold = np.percentile(distances, 95)
            anomalies = np.where(np.array(distances) > anomaly_threshold)[0]
            
            return {
                'anomaly_count': len(anomalies),
                'total_points': len(scaled_data),
                'anomaly_percentage': (len(anomalies) / len(scaled_data)) * 100,
                'method': 'pca_clustering',
                'confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"Erreur détection anomalies multi-variées: {e}")
            return {'anomaly_count': 0, 'error': str(e)}

    async def _analyze_capacity_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les corrélations entre métriques de capacité"""
        
        try:
            pivot_df = df.pivot(index='timestamp', columns='metric', values='value')
            pivot_df = pivot_df.fillna(method='forward').fillna(method='backward')
            
            if len(pivot_df.columns) < 2:
                return {'correlations': {}, 'note': 'insufficient_metrics'}
            
            # Matrice de corrélation
            correlation_matrix = pivot_df.corr()
            
            # Extraction des corrélations significatives
            significant_correlations = {}
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    metric1 = correlation_matrix.columns[i]
                    metric2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.7:  # Corrélation forte
                        significant_correlations[f"{metric1}_vs_{metric2}"] = {
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        }
            
            return {
                'correlations': significant_correlations,
                'correlation_matrix': correlation_matrix.to_dict(),
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse corrélations: {e}")
            return {'correlations': {}, 'error': str(e)}

    async def _generate_performance_visualizations(self, df: pd.DataFrame, trends: Dict[str, Any]) -> List[str]:
        """Génère les visualisations pour le rapport de performance"""
        
        try:
            visualizations = []
            
            # Graphique de tendances temporelles
            fig = make_subplots(
                rows=len(trends), cols=1,
                subplot_titles=list(trends.keys()),
                vertical_spacing=0.05
            )
            
            for i, metric in enumerate(trends.keys(), 1):
                metric_data = df[df['metric'] == metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=metric_data['timestamp'],
                        y=metric_data['value'],
                        mode='lines',
                        name=metric,
                        line=dict(width=2)
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                title="Tendances de Performance - Analyse Temporelle",
                height=300 * len(trends),
                showlegend=False
            )
            
            # Sauvegarde du graphique (en production, sauvegarder dans un storage)
            graph_path = f"performance_trends_{int(datetime.utcnow().timestamp())}.html"
            visualizations.append(graph_path)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Erreur génération visualisations performance: {e}")
            return []

    async def _generate_anomaly_visualizations(self, df: pd.DataFrame, anomalies: Dict[str, Any]) -> List[str]:
        """Génère les visualisations pour le rapport d'anomalies"""
        
        try:
            visualizations = []
            
            # Box plots pour visualiser les anomalies
            fig = make_subplots(
                rows=1, cols=len(anomalies),
                subplot_titles=list(anomalies.keys())
            )
            
            for i, metric in enumerate(anomalies.keys(), 1):
                metric_data = df[df['metric'] == metric]['value']
                
                fig.add_trace(
                    go.Box(
                        y=metric_data,
                        name=metric,
                        boxpoints='outliers'
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(
                title="Détection d'Anomalies - Distribution des Métriques",
                height=400,
                showlegend=False
            )
            
            graph_path = f"anomaly_detection_{int(datetime.utcnow().timestamp())}.html"
            visualizations.append(graph_path)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Erreur génération visualisations anomalies: {e}")
            return []

    async def _generate_capacity_visualizations(self, df: pd.DataFrame, capacity_analysis: Dict[str, Any]) -> List[str]:
        """Génère les visualisations pour le rapport de capacité"""
        
        try:
            visualizations = []
            
            # Graphiques de projection de capacité
            fig = make_subplots(
                rows=len(capacity_analysis), cols=1,
                subplot_titles=[f"{metric} - Projection de Capacité" for metric in capacity_analysis.keys()]
            )
            
            for i, (metric, analysis) in enumerate(capacity_analysis.items(), 1):
                if 'predicted_30d' in analysis:
                    # Données historiques
                    metric_data = df[df['metric'] == metric]
                    
                    # Projection future (simulation)
                    current_value = analysis['current_value']
                    predicted_value = analysis['predicted_30d']
                    
                    # Points de projection
                    future_dates = pd.date_range(
                        start=metric_data['timestamp'].max(),
                        periods=30,
                        freq='D'
                    )
                    
                    projection_values = np.linspace(current_value, predicted_value, 30)
                    
                    # Graphique historique
                    fig.add_trace(
                        go.Scatter(
                            x=metric_data['timestamp'],
                            y=metric_data['value'],
                            mode='lines',
                            name=f"{metric} (Historique)",
                            line=dict(color='blue')
                        ),
                        row=i, col=1
                    )
                    
                    # Graphique projection
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=projection_values,
                            mode='lines',
                            name=f"{metric} (Projection)",
                            line=dict(color='red', dash='dash')
                        ),
                        row=i, col=1
                    )
            
            fig.update_layout(
                title="Planification de Capacité - Projections",
                height=300 * len(capacity_analysis),
                showlegend=True
            )
            
            graph_path = f"capacity_planning_{int(datetime.utcnow().timestamp())}.html"
            visualizations.append(graph_path)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Erreur génération visualisations capacité: {e}")
            return []

    async def generate_comprehensive_report(self, tenant_id: Optional[str] = None) -> AnalyticsReport:
        """Génère un rapport complet multi-analyses"""
        
        try:
            # Génération de tous les types de rapports
            performance_report = None
            anomaly_report = None
            capacity_report = None
            
            for config in self.configs:
                if config.tenant_id == tenant_id:
                    if config.analytics_type == AnalyticsType.PERFORMANCE_TRENDS:
                        performance_report = await self.generate_performance_trends_report(config)
                    elif config.analytics_type == AnalyticsType.ANOMALY_DETECTION:
                        anomaly_report = await self.generate_anomaly_detection_report(config)
                    elif config.analytics_type == AnalyticsType.CAPACITY_PLANNING:
                        capacity_report = await self.generate_capacity_planning_report(config)
            
            # Consolidation des insights et recommandations
            all_insights = []
            all_recommendations = []
            combined_data = {}
            
            for report in [performance_report, anomaly_report, capacity_report]:
                if report:
                    all_insights.extend(report.insights)
                    all_recommendations.extend(report.recommendations)
                    combined_data[report.report_type.value] = report.data
            
            # Insights cross-analyses
            cross_insights = await self._generate_cross_analysis_insights(
                performance_report, anomaly_report, capacity_report
            )
            all_insights.extend(cross_insights)
            
            comprehensive_report = AnalyticsReport(
                report_id=f"comprehensive_{int(datetime.utcnow().timestamp())}",
                report_type=AnalyticsType.BUSINESS_METRICS,
                generated_at=datetime.utcnow(),
                time_range=(datetime.utcnow() - timedelta(days=1), datetime.utcnow()),
                data=combined_data,
                insights=all_insights,
                recommendations=list(set(all_recommendations)),  # Déduplication
                tenant_id=tenant_id,
                confidence_score=0.90
            )
            
            self.reports_history.append(comprehensive_report)
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport complet: {e}")
            raise

    async def _generate_cross_analysis_insights(self, perf_report, anomaly_report, capacity_report) -> List[str]:
        """Génère des insights croisés entre différents types d'analyses"""
        
        insights = []
        
        try:
            # Corrélation entre anomalies et performance
            if anomaly_report and perf_report:
                high_anomaly_metrics = [
                    metric for metric, data in anomaly_report.data.items()
                    if isinstance(data, dict) and data.get('anomaly_percentage', 0) > 5
                ]
                
                declining_metrics = [
                    metric for metric, data in perf_report.data.items()
                    if isinstance(data, dict) and data.get('trend_slope', 0) > 0.1
                ]
                
                common_metrics = set(high_anomaly_metrics) & set(declining_metrics)
                if common_metrics:
                    insights.append(
                        f"🔗 Corrélation détectée: Métriques avec anomalies ET dégradation de performance: {', '.join(common_metrics)}"
                    )
            
            # Prédiction de problèmes futurs
            if capacity_report:
                critical_capacity_metrics = [
                    metric for metric, data in capacity_report.data.items()
                    if isinstance(data, dict) and data.get('time_to_saturation_days', float('inf')) < 60
                ]
                
                if critical_capacity_metrics:
                    insights.append(
                        f"⏰ Risque de saturation dans les 60 prochains jours: {', '.join(critical_capacity_metrics)}"
                    )
            
            # Pattern de charge
            if perf_report:
                high_variance_metrics = [
                    metric for metric, data in perf_report.data.items()
                    if isinstance(data, dict) and data.get('volatility', 0) > data.get('average_value', 0) * 0.3
                ]
                
                if high_variance_metrics:
                    insights.append(
                        f"📊 Métriques avec forte variabilité (charge irrégulière): {', '.join(high_variance_metrics)}"
                    )
            
        except Exception as e:
            logger.error(f"Erreur génération insights croisés: {e}")
        
        return insights

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des analytics"""
        
        total_reports = len(self.reports_history)
        if total_reports == 0:
            return {"total_reports": 0}
        
        recent_reports = [
            r for r in self.reports_history
            if r.generated_at > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Distribution par type
        type_distribution = {}
        for report in self.reports_history:
            report_type = report.report_type.value
            type_distribution[report_type] = type_distribution.get(report_type, 0) + 1
        
        # Score de confiance moyen
        avg_confidence = sum(r.confidence_score for r in self.reports_history) / total_reports
        
        return {
            "total_reports": total_reports,
            "recent_24h": len(recent_reports),
            "average_confidence_score": f"{avg_confidence:.2%}",
            "report_types_distribution": type_distribution,
            "active_configs": len([c for c in self.configs if c.enabled]),
            "last_report_time": max(r.generated_at for r in self.reports_history).isoformat() if self.reports_history else None
        }

# Instance globale du moteur d'analytics
_analytics_engine = AdvancedAnalyticsEngine()

async def generate_analytics_report(analytics_type: AnalyticsType, tenant_id: Optional[str] = None) -> AnalyticsReport:
    """Function helper pour générer un rapport d'analytics"""
    
    if not _analytics_engine.redis_client:
        await _analytics_engine.initialize()
    
    # Trouver la configuration appropriée
    config = next(
        (c for c in _analytics_engine.configs if c.analytics_type == analytics_type and c.tenant_id == tenant_id),
        None
    )
    
    if not config:
        raise ValueError(f"Configuration non trouvée pour {analytics_type.value}")
    
    if analytics_type == AnalyticsType.PERFORMANCE_TRENDS:
        return await _analytics_engine.generate_performance_trends_report(config)
    elif analytics_type == AnalyticsType.ANOMALY_DETECTION:
        return await _analytics_engine.generate_anomaly_detection_report(config)
    elif analytics_type == AnalyticsType.CAPACITY_PLANNING:
        return await _analytics_engine.generate_capacity_planning_report(config)
    else:
        raise ValueError(f"Type d'analyse non supporté: {analytics_type.value}")

async def get_analytics_engine() -> AdvancedAnalyticsEngine:
    """Retourne l'instance du moteur d'analytics"""
    return _analytics_engine

# Configuration des alertes d'analytics
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    analytics_configs = [
        AlertConfig(
            name="analytics_performance_degradation",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.ANALYTICS,
            conditions=['Dégradation de performance détectée par analytics'],
            actions=['generate_performance_report', 'notify_engineering_team'],
            ml_enabled=True
        ),
        AlertConfig(
            name="analytics_capacity_warning",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.MEDIUM,
            script_type=ScriptType.ANALYTICS,
            conditions=['Saturation de capacité prévue'],
            actions=['generate_capacity_report', 'plan_scaling'],
            ml_enabled=True
        )
    ]
    
    for config in analytics_configs:
        register_alert(config)
