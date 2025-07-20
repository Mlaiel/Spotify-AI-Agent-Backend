#!/usr/bin/env python3
"""
Advanced Analytics and Reporting Engine
======================================

Moteur d'analyse et de reporting avancé pour les métriques tenancy
avec ML, prédictions, tableaux de bord et alertes intelligentes.
"""

import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Imports pour ML et analytics
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import des schémas
from .tenant_config_schema import TenantConfigSchema, TenantType
from .monitoring_schema import MonitoringConfigSchema, MonitoringMetric
from .performance_schema import PerformanceMetricsSchema, AnomalyDetection


class AnalyticsType(Enum):
    """Types d'analyses disponibles."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class ReportFormat(Enum):
    """Formats de rapport disponibles."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    DASHBOARD = "dashboard"


@dataclass
class MetricSummary:
    """Résumé statistique d'une métrique."""
    metric_name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    anomaly_score: float  # 0.0 to 1.0
    last_updated: datetime


@dataclass
class TrendAnalysis:
    """Analyse de tendance pour une métrique."""
    metric_name: str
    time_period: str
    trend_type: str  # "linear", "exponential", "seasonal", "irregular"
    slope: float
    r_squared: float
    forecast_points: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]
    seasonality_detected: bool
    seasonal_period: Optional[int] = None


@dataclass
class AnomalyReport:
    """Rapport d'anomalie détecté."""
    metric_name: str
    timestamp: datetime
    actual_value: float
    expected_value: float
    deviation_score: float  # Nombre d'écarts-types
    anomaly_type: str  # "spike", "drop", "drift", "outlier"
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any]


@dataclass
class PerformanceInsight:
    """Insight de performance avec recommandations."""
    tenant_id: str
    insight_type: str
    title: str
    description: str
    impact: str  # "low", "medium", "high", "critical"
    recommendation: str
    estimated_improvement: Optional[str] = None
    implementation_effort: Optional[str] = None  # "low", "medium", "high"
    priority_score: float = 0.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Classe de base pour les analyseurs."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Méthode d'analyse principale."""
        pass


class DescriptiveAnalyzer(BaseAnalyzer):
    """Analyseur pour les statistiques descriptives."""
    
    def __init__(self):
        super().__init__("descriptive")
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les statistiques descriptives."""
        
        results = {
            "metric_summaries": [],
            "correlation_matrix": {},
            "distribution_analysis": {},
            "temporal_patterns": {}
        }
        
        # Convertir les données en DataFrame si possible
        if not data.get("metrics_data"):
            return results
        
        try:
            df = pd.DataFrame(data["metrics_data"])
            
            # Calculer les résumés par métrique
            for column in df.select_dtypes(include=[np.number]).columns:
                series = df[column].dropna()
                if len(series) > 0:
                    summary = MetricSummary(
                        metric_name=column,
                        count=len(series),
                        mean=float(series.mean()),
                        std=float(series.std()),
                        min=float(series.min()),
                        max=float(series.max()),
                        percentile_25=float(series.quantile(0.25)),
                        percentile_50=float(series.quantile(0.50)),
                        percentile_75=float(series.quantile(0.75)),
                        percentile_95=float(series.quantile(0.95)),
                        percentile_99=float(series.quantile(0.99)),
                        trend_direction=self._calculate_trend_direction(series),
                        trend_strength=self._calculate_trend_strength(series),
                        anomaly_score=self._calculate_anomaly_score(series),
                        last_updated=datetime.now(timezone.utc)
                    )
                    results["metric_summaries"].append(summary.__dict__)
            
            # Matrice de corrélation
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                results["correlation_matrix"] = corr_matrix.to_dict()
            
            # Analyse temporelle si timestamp disponible
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                results["temporal_patterns"] = self._analyze_temporal_patterns(df)
        
        except Exception as e:
            self.logger.error(f"Error in descriptive analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calcule la direction de la tendance."""
        if len(series) < 2:
            return "stable"
        
        # Calcul de la pente avec régression linéaire simple
        x = np.arange(len(series))
        y = series.values
        
        # Enlever les valeurs NaN
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return "stable"
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calcule la force de la tendance (R²)."""
        if len(series) < 3:
            return 0.0
        
        try:
            x = np.arange(len(series))
            y = series.values
            
            # Enlever les valeurs NaN
            mask = ~np.isnan(y)
            if np.sum(mask) < 3:
                return 0.0
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Calculer R²
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            y_pred = slope * x_clean + intercept
            
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            return max(0.0, min(1.0, r_squared))
        
        except Exception:
            return 0.0
    
    def _calculate_anomaly_score(self, series: pd.Series) -> float:
        """Calcule un score d'anomalie basé sur les outliers."""
        if len(series) < 5:
            return 0.0
        
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            anomaly_ratio = len(outliers) / len(series)
            
            return min(1.0, anomaly_ratio * 2)  # Normaliser entre 0 et 1
        
        except Exception:
            return 0.0
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les patterns temporels."""
        patterns = {
            "hourly_patterns": {},
            "daily_patterns": {},
            "weekly_patterns": {},
            "monthly_patterns": {}
        }
        
        try:
            df_sorted = df.sort_values("timestamp")
            
            # Patterns horaires
            df_sorted["hour"] = df_sorted["timestamp"].dt.hour
            hourly_agg = df_sorted.groupby("hour").agg({
                col: ["mean", "std", "count"] 
                for col in df_sorted.select_dtypes(include=[np.number]).columns 
                if col != "hour"
            })
            patterns["hourly_patterns"] = hourly_agg.to_dict()
            
            # Patterns journaliers
            df_sorted["day_of_week"] = df_sorted["timestamp"].dt.dayofweek
            daily_agg = df_sorted.groupby("day_of_week").agg({
                col: ["mean", "std", "count"] 
                for col in df_sorted.select_dtypes(include=[np.number]).columns 
                if col not in ["hour", "day_of_week"]
            })
            patterns["daily_patterns"] = daily_agg.to_dict()
        
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            patterns["error"] = str(e)
        
        return patterns


class PredictiveAnalyzer(BaseAnalyzer):
    """Analyseur pour les prédictions ML."""
    
    def __init__(self):
        super().__init__("predictive")
        self.models = {}
        self.scalers = {}
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue des analyses prédictives."""
        
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available for ML predictions"}
        
        results = {
            "trend_forecasts": [],
            "anomaly_predictions": [],
            "capacity_planning": {},
            "performance_predictions": {}
        }
        
        try:
            if not data.get("metrics_data"):
                return results
            
            df = pd.DataFrame(data["metrics_data"])
            
            # Prédictions de tendances
            for column in df.select_dtypes(include=[np.number]).columns:
                series = df[column].dropna()
                if len(series) > 10:  # Minimum de données pour prédiction
                    trend_forecast = await self._forecast_trend(column, series)
                    if trend_forecast:
                        results["trend_forecasts"].append(trend_forecast)
            
            # Détection d'anomalies avec ML
            anomaly_results = await self._predict_anomalies(df)
            results["anomaly_predictions"] = anomaly_results
            
            # Planification de capacité
            capacity_results = await self._analyze_capacity(df)
            results["capacity_planning"] = capacity_results
        
        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _forecast_trend(self, metric_name: str, series: pd.Series, 
                            forecast_periods: int = 24) -> Optional[Dict[str, Any]]:
        """Prévoit la tendance d'une métrique."""
        
        try:
            # Préparer les données
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            # Enlever les NaN
            mask = ~np.isnan(y)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                return None
            
            # Modèle de régression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_clean, y_clean)
            
            # Prédictions futures
            future_X = np.arange(len(series), len(series) + forecast_periods).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Calcul de l'incertitude (approximation)
            # En production, utiliser des modèles avec intervalles de confiance
            std_error = np.std(y_clean - model.predict(X_clean))
            
            forecast_points = []
            confidence_intervals = []
            
            for i, pred in enumerate(predictions):
                # Timestamp futur (assumant des intervalles d'1 heure)
                future_time = datetime.now(timezone.utc) + timedelta(hours=i+1)
                forecast_points.append((future_time.isoformat(), float(pred)))
                
                # Intervalle de confiance simple
                lower = pred - 1.96 * std_error
                upper = pred + 1.96 * std_error
                confidence_intervals.append((float(lower), float(upper)))
            
            return {
                "metric_name": metric_name,
                "time_period": f"{forecast_periods}h",
                "trend_type": "ml_regression",
                "forecast_points": forecast_points,
                "confidence_intervals": confidence_intervals,
                "model_score": float(model.score(X_clean, y_clean)),
                "std_error": float(std_error)
            }
        
        except Exception as e:
            self.logger.error(f"Error forecasting {metric_name}: {e}")
            return None
    
    async def _predict_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prédit les anomalies avec Isolation Forest."""
        
        anomalies = []
        
        try:
            # Sélectionner les colonnes numériques
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) == 0 or len(numeric_df) < 20:
                return anomalies
            
            # Normaliser les données
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.mean()))
            
            # Modèle Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.decision_function(scaled_data)
            
            # Identifier les anomalies
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            for idx in anomaly_indices:
                if idx < len(df):
                    anomaly_data = {
                        "timestamp": df.iloc[idx].get("timestamp", datetime.now(timezone.utc)).isoformat(),
                        "anomaly_score": float(anomaly_scores[idx]),
                        "severity": self._classify_anomaly_severity(anomaly_scores[idx]),
                        "affected_metrics": {},
                        "confidence": min(1.0, abs(float(anomaly_scores[idx])) / 0.5)
                    }
                    
                    # Ajouter les valeurs des métriques affectées
                    for col in numeric_df.columns:
                        if pd.notna(df.iloc[idx][col]):
                            anomaly_data["affected_metrics"][col] = float(df.iloc[idx][col])
                    
                    anomalies.append(anomaly_data)
        
        except Exception as e:
            self.logger.error(f"Error in anomaly prediction: {e}")
        
        return anomalies
    
    def _classify_anomaly_severity(self, score: float) -> str:
        """Classifie la sévérité d'une anomalie."""
        abs_score = abs(score)
        
        if abs_score >= 0.7:
            return "critical"
        elif abs_score >= 0.5:
            return "high"
        elif abs_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    async def _analyze_capacity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse la planification de capacité."""
        
        capacity_analysis = {
            "resource_utilization": {},
            "growth_projections": {},
            "bottleneck_predictions": {},
            "recommendations": []
        }
        
        try:
            # Analyse d'utilisation des ressources
            resource_metrics = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]
            
            for metric in resource_metrics:
                if metric in df.columns:
                    series = df[metric].dropna()
                    if len(series) > 0:
                        current_usage = float(series.iloc[-1]) if len(series) > 0 else 0.0
                        avg_usage = float(series.mean())
                        max_usage = float(series.max())
                        trend = self._calculate_trend_direction(series)
                        
                        capacity_analysis["resource_utilization"][metric] = {
                            "current_usage": current_usage,
                            "average_usage": avg_usage,
                            "peak_usage": max_usage,
                            "trend": trend,
                            "utilization_level": self._classify_utilization_level(current_usage)
                        }
                        
                        # Prédiction de saturation
                        if trend == "increasing" and current_usage > 70:
                            days_to_saturation = self._estimate_saturation_time(series)
                            if days_to_saturation:
                                capacity_analysis["bottleneck_predictions"][metric] = {
                                    "estimated_days_to_saturation": days_to_saturation,
                                    "confidence": "medium"
                                }
                                
                                capacity_analysis["recommendations"].append({
                                    "type": "capacity_scaling",
                                    "metric": metric,
                                    "recommendation": f"Consider scaling {metric.replace('_', ' ')} capacity within {days_to_saturation} days",
                                    "urgency": "high" if days_to_saturation < 7 else "medium"
                                })
        
        except Exception as e:
            self.logger.error(f"Error in capacity analysis: {e}")
            capacity_analysis["error"] = str(e)
        
        return capacity_analysis
    
    def _classify_utilization_level(self, usage: float) -> str:
        """Classifie le niveau d'utilisation."""
        if usage >= 90:
            return "critical"
        elif usage >= 80:
            return "high"
        elif usage >= 60:
            return "medium"
        else:
            return "normal"
    
    def _estimate_saturation_time(self, series: pd.Series) -> Optional[int]:
        """Estime le temps avant saturation (en jours)."""
        if len(series) < 5:
            return None
        
        try:
            # Régression linéaire simple pour estimer la tendance
            x = np.arange(len(series))
            y = series.values
            
            # Enlever les NaN
            mask = ~np.isnan(y)
            if np.sum(mask) < 3:
                return None
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            
            if slope <= 0:
                return None
            
            # Calculer quand ça atteindra 100%
            current_value = y_clean[-1]
            periods_to_100 = (100 - current_value) / slope
            
            # Convertir en jours (assumant des mesures horaires)
            days_to_saturation = int(periods_to_100 / 24)
            
            return max(1, days_to_saturation) if days_to_saturation > 0 else None
        
        except Exception:
            return None


class InsightGenerator:
    """Générateur d'insights et recommandations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_insights(self, analytics_results: Dict[str, Any], 
                              tenant_config: Optional[TenantConfigSchema] = None) -> List[PerformanceInsight]:
        """Génère des insights basés sur les résultats d'analyse."""
        
        insights = []
        
        try:
            # Insights basés sur les statistiques descriptives
            if "descriptive" in analytics_results:
                descriptive_insights = await self._generate_descriptive_insights(
                    analytics_results["descriptive"], tenant_config
                )
                insights.extend(descriptive_insights)
            
            # Insights basés sur les prédictions
            if "predictive" in analytics_results:
                predictive_insights = await self._generate_predictive_insights(
                    analytics_results["predictive"], tenant_config
                )
                insights.extend(predictive_insights)
            
            # Insights de compliance et sécurité
            compliance_insights = await self._generate_compliance_insights(tenant_config)
            insights.extend(compliance_insights)
            
            # Trier par priorité
            insights.sort(key=lambda x: x.priority_score, reverse=True)
        
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        return insights
    
    async def _generate_descriptive_insights(self, descriptive_data: Dict[str, Any], 
                                           tenant_config: Optional[TenantConfigSchema]) -> List[PerformanceInsight]:
        """Génère des insights basés sur les statistiques descriptives."""
        
        insights = []
        tenant_id = tenant_config.tenant_id if tenant_config else "unknown"
        
        try:
            metric_summaries = descriptive_data.get("metric_summaries", [])
            
            for summary in metric_summaries:
                metric_name = summary.get("metric_name", "")
                
                # Insight sur les métriques avec forte variabilité
                if summary.get("std", 0) > summary.get("mean", 0) * 0.5:
                    insights.append(PerformanceInsight(
                        tenant_id=tenant_id,
                        insight_type="variability",
                        title=f"High Variability in {metric_name}",
                        description=f"The metric {metric_name} shows high variability (CV={summary.get('std', 0)/summary.get('mean', 1):.2f}), which may indicate instability.",
                        impact="medium",
                        recommendation="Investigate root causes of variability and implement stabilization measures.",
                        priority_score=0.7,
                        supporting_data={
                            "coefficient_of_variation": summary.get("std", 0) / summary.get("mean", 1),
                            "metric_stats": summary
                        }
                    ))
                
                # Insight sur les anomalies
                if summary.get("anomaly_score", 0) > 0.3:
                    severity = "high" if summary.get("anomaly_score", 0) > 0.7 else "medium"
                    insights.append(PerformanceInsight(
                        tenant_id=tenant_id,
                        insight_type="anomaly",
                        title=f"Anomalies Detected in {metric_name}",
                        description=f"Significant anomalies detected in {metric_name} (score: {summary.get('anomaly_score', 0):.2f}).",
                        impact=severity,
                        recommendation="Review recent changes and investigate potential causes of anomalous behavior.",
                        priority_score=0.8 if severity == "high" else 0.6,
                        supporting_data={
                            "anomaly_score": summary.get("anomaly_score", 0),
                            "metric_stats": summary
                        }
                    ))
                
                # Insight sur les tendances négatives
                if summary.get("trend_direction") == "decreasing" and "response_time" in metric_name.lower():
                    insights.append(PerformanceInsight(
                        tenant_id=tenant_id,
                        insight_type="performance_improvement",
                        title=f"Improving Response Times in {metric_name}",
                        description=f"Response times are showing a positive decreasing trend.",
                        impact="low",
                        recommendation="Continue current optimization efforts and document successful practices.",
                        priority_score=0.3,
                        supporting_data={
                            "trend_direction": summary.get("trend_direction"),
                            "trend_strength": summary.get("trend_strength", 0)
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Error generating descriptive insights: {e}")
        
        return insights
    
    async def _generate_predictive_insights(self, predictive_data: Dict[str, Any], 
                                          tenant_config: Optional[TenantConfigSchema]) -> List[PerformanceInsight]:
        """Génère des insights basés sur les prédictions."""
        
        insights = []
        tenant_id = tenant_config.tenant_id if tenant_config else "unknown"
        
        try:
            # Insights sur les prédictions d'anomalies
            anomaly_predictions = predictive_data.get("anomaly_predictions", [])
            if anomaly_predictions:
                critical_anomalies = [a for a in anomaly_predictions if a.get("severity") == "critical"]
                
                if critical_anomalies:
                    insights.append(PerformanceInsight(
                        tenant_id=tenant_id,
                        insight_type="anomaly_prediction",
                        title="Critical Anomalies Predicted",
                        description=f"ML models predict {len(critical_anomalies)} critical anomalies in system metrics.",
                        impact="critical",
                        recommendation="Immediate investigation required. Review system health and recent changes.",
                        priority_score=0.95,
                        supporting_data={
                            "critical_anomalies_count": len(critical_anomalies),
                            "anomalies": critical_anomalies[:5]  # Limiter pour éviter les gros objets
                        }
                    ))
            
            # Insights sur la planification de capacité
            capacity_planning = predictive_data.get("capacity_planning", {})
            bottlenecks = capacity_planning.get("bottleneck_predictions", {})
            
            for metric, prediction in bottlenecks.items():
                days_to_saturation = prediction.get("estimated_days_to_saturation", 0)
                
                if days_to_saturation <= 7:
                    impact = "critical"
                    priority = 0.9
                elif days_to_saturation <= 30:
                    impact = "high"
                    priority = 0.7
                else:
                    impact = "medium"
                    priority = 0.5
                
                insights.append(PerformanceInsight(
                    tenant_id=tenant_id,
                    insight_type="capacity_planning",
                    title=f"Capacity Bottleneck Predicted for {metric}",
                    description=f"Current trends suggest {metric} will reach capacity limits in approximately {days_to_saturation} days.",
                    impact=impact,
                    recommendation=f"Plan capacity scaling for {metric} within {max(1, days_to_saturation-1)} days to prevent service degradation.",
                    estimated_improvement="Prevent service outages and maintain SLA compliance",
                    implementation_effort="medium",
                    priority_score=priority,
                    supporting_data={
                        "days_to_saturation": days_to_saturation,
                        "metric_name": metric,
                        "confidence": prediction.get("confidence", "unknown")
                    }
                ))
        
        except Exception as e:
            self.logger.error(f"Error generating predictive insights: {e}")
        
        return insights
    
    async def _generate_compliance_insights(self, tenant_config: Optional[TenantConfigSchema]) -> List[PerformanceInsight]:
        """Génère des insights liés à la compliance."""
        
        insights = []
        
        if not tenant_config:
            return insights
        
        try:
            tenant_id = tenant_config.tenant_id
            
            # Vérifier les niveaux de compliance
            compliance_levels = getattr(tenant_config, 'compliance_levels', [])
            
            if not compliance_levels:
                insights.append(PerformanceInsight(
                    tenant_id=tenant_id,
                    insight_type="compliance",
                    title="No Compliance Standards Configured",
                    description="No compliance standards are currently configured for this tenant.",
                    impact="medium",
                    recommendation="Configure appropriate compliance standards (GDPR, SOC2, etc.) based on business requirements.",
                    priority_score=0.6,
                    supporting_data={
                        "current_compliance_levels": compliance_levels
                    }
                ))
            
            # Vérifier les SLA selon le type de tenant
            if hasattr(tenant_config, 'sla') and tenant_config.sla:
                sla = tenant_config.sla
                uptime = getattr(sla, 'uptime_percentage', 0)
                
                if tenant_config.tenant_type == TenantType.ENTERPRISE and uptime < 99.9:
                    insights.append(PerformanceInsight(
                        tenant_id=tenant_id,
                        insight_type="sla_compliance",
                        title="Enterprise SLA Below Industry Standard",
                        description=f"Current SLA uptime ({uptime}%) is below enterprise standards (99.9%+).",
                        impact="high",
                        recommendation="Upgrade infrastructure and processes to meet enterprise SLA requirements.",
                        estimated_improvement="Improved customer satisfaction and contract compliance",
                        implementation_effort="high",
                        priority_score=0.8,
                        supporting_data={
                            "current_uptime": uptime,
                            "recommended_uptime": 99.9,
                            "tenant_type": tenant_config.tenant_type.value
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Error generating compliance insights: {e}")
        
        return insights


class ReportGenerator:
    """Générateur de rapports multi-format."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_report(self, analytics_results: Dict[str, Any], 
                            insights: List[PerformanceInsight],
                            format: ReportFormat = ReportFormat.JSON,
                            output_path: Optional[Path] = None) -> Union[str, Dict[str, Any]]:
        """Génère un rapport dans le format spécifié."""
        
        if format == ReportFormat.JSON:
            return await self._generate_json_report(analytics_results, insights, output_path)
        elif format == ReportFormat.HTML:
            return await self._generate_html_report(analytics_results, insights, output_path)
        elif format == ReportFormat.DASHBOARD:
            return await self._generate_dashboard(analytics_results, insights)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    async def _generate_json_report(self, analytics_results: Dict[str, Any], 
                                  insights: List[PerformanceInsight],
                                  output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Génère un rapport JSON."""
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_type": "analytics_and_insights",
                "version": "1.0.0"
            },
            "analytics_results": analytics_results,
            "insights": [insight.__dict__ for insight in insights],
            "summary": {
                "total_insights": len(insights),
                "critical_insights": len([i for i in insights if i.impact == "critical"]),
                "high_priority_insights": len([i for i in insights if i.priority_score > 0.7]),
                "insight_types": list(set([i.insight_type for i in insights]))
            }
        }
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    async def _generate_html_report(self, analytics_results: Dict[str, Any], 
                                  insights: List[PerformanceInsight],
                                  output_path: Optional[Path] = None) -> str:
        """Génère un rapport HTML."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tenancy Analytics Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .insight { margin: 20px 0; padding: 15px; border-left: 4px solid #007ACC; background-color: #f9f9f9; }
                .critical { border-left-color: #d32f2f; }
                .high { border-left-color: #f57c00; }
                .medium { border-left-color: #fbc02d; }
                .low { border-left-color: #388e3c; }
                .metric-summary { background-color: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .recommendation { background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Tenancy Analytics Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Total Insights: {total_insights} | Critical: {critical_insights} | High Priority: {high_priority_insights}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="metric-summary">
                <p>This report provides comprehensive analytics and actionable insights for tenant performance, capacity planning, and system optimization.</p>
                <ul>
                    <li><strong>Analytics Coverage:</strong> {analytics_types}</li>
                    <li><strong>Insight Categories:</strong> {insight_types}</li>
                    <li><strong>Recommendations:</strong> {total_insights} actionable items identified</li>
                </ul>
            </div>
            
            <h2>Key Insights & Recommendations</h2>
            {insights_html}
            
            <h2>Detailed Analytics</h2>
            <div class="metric-summary">
                <h3>Descriptive Statistics</h3>
                <pre>{descriptive_stats}</pre>
            </div>
            
            <div class="metric-summary">
                <h3>Predictive Analysis</h3>
                <pre>{predictive_stats}</pre>
            </div>
        </body>
        </html>
        """
        
        # Préparer les insights HTML
        insights_html = ""
        for insight in insights:
            impact_class = insight.impact.lower()
            insights_html += f"""
            <div class="insight {impact_class}">
                <h3>{insight.title}</h3>
                <p><strong>Impact:</strong> {insight.impact.title()} | <strong>Priority:</strong> {insight.priority_score:.2f}</p>
                <p>{insight.description}</p>
                <div class="recommendation">
                    <strong>Recommendation:</strong> {insight.recommendation}
                </div>
                {f'<p><strong>Estimated Improvement:</strong> {insight.estimated_improvement}</p>' if insight.estimated_improvement else ''}
                {f'<p><strong>Implementation Effort:</strong> {insight.implementation_effort}</p>' if insight.implementation_effort else ''}
            </div>
            """
        
        # Préparer les données pour le template
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        total_insights = len(insights)
        critical_insights = len([i for i in insights if i.impact == "critical"])
        high_priority_insights = len([i for i in insights if i.priority_score > 0.7])
        analytics_types = ", ".join(analytics_results.keys())
        insight_types = ", ".join(set([i.insight_type for i in insights]))
        
        descriptive_stats = json.dumps(analytics_results.get("descriptive", {}), indent=2)
        predictive_stats = json.dumps(analytics_results.get("predictive", {}), indent=2)
        
        # Générer le HTML final
        html_content = html_template.format(
            timestamp=timestamp,
            total_insights=total_insights,
            critical_insights=critical_insights,
            high_priority_insights=high_priority_insights,
            analytics_types=analytics_types,
            insight_types=insight_types,
            insights_html=insights_html,
            descriptive_stats=descriptive_stats,
            predictive_stats=predictive_stats
        )
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    async def _generate_dashboard(self, analytics_results: Dict[str, Any], 
                                insights: List[PerformanceInsight]) -> Dict[str, Any]:
        """Génère les données pour un tableau de bord interactif."""
        
        if not HAS_PLOTLY:
            return {"error": "Plotly not available for dashboard generation"}
        
        dashboard_data = {
            "charts": [],
            "kpis": [],
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # KPIs principaux
            descriptive = analytics_results.get("descriptive", {})
            metric_summaries = descriptive.get("metric_summaries", [])
            
            for summary in metric_summaries:
                metric_name = summary.get("metric_name", "")
                current_value = summary.get("mean", 0)
                trend = summary.get("trend_direction", "stable")
                
                dashboard_data["kpis"].append({
                    "name": metric_name.replace("_", " ").title(),
                    "value": current_value,
                    "trend": trend,
                    "unit": self._infer_unit(metric_name)
                })
            
            # Alertes critiques
            critical_insights = [i for i in insights if i.impact == "critical"]
            for insight in critical_insights:
                dashboard_data["alerts"].append({
                    "title": insight.title,
                    "description": insight.description,
                    "priority": insight.priority_score,
                    "type": insight.insight_type
                })
            
            # Recommandations top 5
            top_recommendations = sorted(insights, key=lambda x: x.priority_score, reverse=True)[:5]
            for rec in top_recommendations:
                dashboard_data["recommendations"].append({
                    "title": rec.title,
                    "description": rec.recommendation,
                    "impact": rec.impact,
                    "effort": rec.implementation_effort or "unknown",
                    "priority": rec.priority_score
                })
        
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {e}")
            dashboard_data["error"] = str(e)
        
        return dashboard_data
    
    def _infer_unit(self, metric_name: str) -> str:
        """Infère l'unité d'une métrique basée sur son nom."""
        name_lower = metric_name.lower()
        
        if "percent" in name_lower or "rate" in name_lower:
            return "%"
        elif "time" in name_lower and "ms" not in name_lower:
            return "ms"
        elif "bytes" in name_lower:
            return "bytes"
        elif "count" in name_lower or "total" in name_lower:
            return "count"
        else:
            return ""


class AnalyticsEngine:
    """Moteur principal d'analyse et de reporting."""
    
    def __init__(self):
        self.descriptive_analyzer = DescriptiveAnalyzer()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.insight_generator = InsightGenerator()
        self.report_generator = ReportGenerator()
        self.logger = logging.getLogger(__name__)
    
    async def run_complete_analysis(self, metrics_data: List[Dict[str, Any]], 
                                  tenant_config: Optional[TenantConfigSchema] = None,
                                  analysis_types: List[AnalyticsType] = None) -> Dict[str, Any]:
        """Exécute une analyse complète."""
        
        if analysis_types is None:
            analysis_types = [AnalyticsType.DESCRIPTIVE, AnalyticsType.PREDICTIVE]
        
        results = {
            "analysis_metadata": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_config.tenant_id if tenant_config else "unknown",
                "analysis_types": [t.value for t in analysis_types],
                "data_points": len(metrics_data)
            },
            "analytics": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            data_input = {"metrics_data": metrics_data}
            
            # Analyses descriptives
            if AnalyticsType.DESCRIPTIVE in analysis_types:
                self.logger.info("Running descriptive analysis...")
                descriptive_results = await self.descriptive_analyzer.analyze(data_input)
                results["analytics"]["descriptive"] = descriptive_results
            
            # Analyses prédictives
            if AnalyticsType.PREDICTIVE in analysis_types:
                self.logger.info("Running predictive analysis...")
                predictive_results = await self.predictive_analyzer.analyze(data_input)
                results["analytics"]["predictive"] = predictive_results
            
            # Génération d'insights
            self.logger.info("Generating insights...")
            insights = await self.insight_generator.generate_insights(
                results["analytics"], tenant_config
            )
            results["insights"] = [insight.__dict__ for insight in insights]
            
            # Recommandations prioritaires
            high_priority_insights = [i for i in insights if i.priority_score > 0.7]
            results["recommendations"] = [
                {
                    "title": insight.title,
                    "recommendation": insight.recommendation,
                    "priority": insight.priority_score,
                    "impact": insight.impact
                }
                for insight in high_priority_insights
            ]
            
            results["analysis_metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
            results["analysis_metadata"]["total_insights"] = len(insights)
            results["analysis_metadata"]["high_priority_insights"] = len(high_priority_insights)
        
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    async def generate_reports(self, analysis_results: Dict[str, Any],
                             formats: List[ReportFormat] = None,
                             output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Génère des rapports dans multiple formats."""
        
        if formats is None:
            formats = [ReportFormat.JSON, ReportFormat.HTML]
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        insights = [PerformanceInsight(**insight) for insight in analysis_results.get("insights", [])]
        
        try:
            for format in formats:
                if format == ReportFormat.JSON:
                    output_path = output_dir / "analytics_report.json" if output_dir else None
                    report = await self.report_generator.generate_report(
                        analysis_results["analytics"], insights, format, output_path
                    )
                    reports["json"] = report
                
                elif format == ReportFormat.HTML:
                    output_path = output_dir / "analytics_report.html" if output_dir else None
                    report = await self.report_generator.generate_report(
                        analysis_results["analytics"], insights, format, output_path
                    )
                    reports["html"] = report
                
                elif format == ReportFormat.DASHBOARD:
                    report = await self.report_generator.generate_report(
                        analysis_results["analytics"], insights, format
                    )
                    reports["dashboard"] = report
        
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            reports["error"] = str(e)
        
        return reports


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation complète du moteur d'analyse."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Données d'exemple
    sample_metrics_data = [
        {
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=i),
            "cpu_usage_percent": 45 + np.random.normal(0, 5) + i * 0.5,  # Tendance croissante
            "memory_usage_percent": 60 + np.random.normal(0, 3),
            "response_time_ms": 250 + np.random.normal(0, 50),
            "error_rate_percent": max(0, 2 + np.random.normal(0, 1))
        }
        for i in range(168)  # 1 semaine de données horaires
    ]
    
    # Configuration tenant d'exemple
    from .tenant_config_schema import TenantType
    tenant_config = TenantConfigSchema(
        tenant_id="analytics_demo_001",
        tenant_name="Analytics Demo Corporation",
        tenant_type=TenantType.ENTERPRISE,
        admin_email="admin@analytics-demo.com",
        compliance_levels=["gdpr", "soc2"]
    )
    
    # Créer le moteur d'analyse
    engine = AnalyticsEngine()
    
    # Exécuter l'analyse complète
    print("🔍 Running complete analytics...")
    analysis_results = await engine.run_complete_analysis(
        sample_metrics_data, 
        tenant_config,
        [AnalyticsType.DESCRIPTIVE, AnalyticsType.PREDICTIVE]
    )
    
    # Générer des rapports
    print("📊 Generating reports...")
    reports = await engine.generate_reports(
        analysis_results,
        [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.DASHBOARD],
        Path("/tmp/analytics_reports")
    )
    
    print("✅ Analysis complete!")
    print(f"📈 Generated {len(analysis_results.get('insights', []))} insights")
    print(f"🎯 {len(analysis_results.get('recommendations', []))} high-priority recommendations")
    print(f"📄 Reports available in: {list(reports.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
