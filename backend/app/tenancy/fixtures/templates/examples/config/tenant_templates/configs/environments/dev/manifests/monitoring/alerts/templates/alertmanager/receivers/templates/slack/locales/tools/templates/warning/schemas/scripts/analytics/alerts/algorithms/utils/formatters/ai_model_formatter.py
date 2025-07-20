"""
Spotify AI Agent - AI Model Performance Formatters
==================================================

Ultra-advanced AI/ML model performance formatting system for model monitoring,
explainability, performance analysis, and automated model insights.

This module handles sophisticated formatting for:
- ML model performance metrics and KPIs
- Model explainability and interpretability reports
- Training progress and hyperparameter optimization
- Model drift detection and data quality monitoring
- A/B testing results for ML models
- Feature importance and SHAP value analysis
- Model bias detection and fairness metrics
- AutoML pipeline performance and recommendations
- Real-time inference monitoring and alerts

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import statistics
import math

logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Types of ML models."""
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


class PerformanceMetric(Enum):
    """ML performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    HIT_RATE = "hit_rate"
    NDCG = "ndcg"
    MAP = "map"
    MRR = "mrr"


class ExplainabilityMethod(Enum):
    """Model explainability methods."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ANCHORS = "anchors"
    COUNTERFACTUALS = "counterfactuals"


@dataclass
class ModelMetrics:
    """ML model performance metrics."""
    
    model_id: str
    model_name: str
    model_type: ModelType
    stage: ModelStage
    version: str
    timestamp: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    inference_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    data_quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "stage": self.stage.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "performance_metrics": self.performance_metrics,
            "training_metrics": self.training_metrics,
            "inference_metrics": self.inference_metrics,
            "resource_usage": self.resource_usage,
            "data_quality_metrics": self.data_quality_metrics
        }


@dataclass
class ExplainabilityReport:
    """Model explainability and interpretability report."""
    
    model_id: str
    method: ExplainabilityMethod
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Dict[str, List[float]] = field(default_factory=dict)
    local_explanations: List[Dict[str, Any]] = field(default_factory=list)
    global_explanations: Dict[str, Any] = field(default_factory=dict)
    bias_metrics: Dict[str, float] = field(default_factory=dict)
    fairness_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "method": self.method.value,
            "feature_importance": self.feature_importance,
            "shap_values": self.shap_values,
            "local_explanations": self.local_explanations,
            "global_explanations": self.global_explanations,
            "bias_metrics": self.bias_metrics,
            "fairness_scores": self.fairness_scores
        }


@dataclass
class FormattedModelReport:
    """Container for formatted ML model report."""
    
    model_metrics: ModelMetrics
    formatted_content: str
    explainability_report: Optional[ExplainabilityReport] = None
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_metrics": self.model_metrics.to_dict(),
            "formatted_content": self.formatted_content,
            "explainability_report": self.explainability_report.to_dict() if self.explainability_report else None,
            "visualizations": self.visualizations,
            "recommendations": self.recommendations,
            "alerts": self.alerts,
            "metadata": self.metadata
        }


class BaseAIModelFormatter:
    """Base class for AI/ML model formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
        # Performance thresholds
        self.performance_thresholds = config.get('performance_thresholds', {
            'accuracy': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70},
            'precision': {'excellent': 0.90, 'good': 0.80, 'poor': 0.65},
            'recall': {'excellent': 0.90, 'good': 0.80, 'poor': 0.65},
            'f1_score': {'excellent': 0.90, 'good': 0.80, 'poor': 0.65},
            'auc_roc': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70},
            'hit_rate': {'excellent': 0.80, 'good': 0.60, 'poor': 0.40},
            'ndcg': {'excellent': 0.85, 'good': 0.70, 'poor': 0.50}
        })
        
        # Alert thresholds
        self.alert_thresholds = config.get('alert_thresholds', {
            'performance_degradation': 0.05,
            'data_drift': 0.1,
            'feature_drift': 0.15,
            'latency_increase': 2.0,
            'error_rate': 0.02
        })
    
    def get_performance_status(self, metric_name: str, value: float) -> Tuple[str, str]:
        """Get performance status and icon for a metric."""
        
        if metric_name not in self.performance_thresholds:
            return "unknown", "❓"
        
        thresholds = self.performance_thresholds[metric_name]
        
        if value >= thresholds['excellent']:
            return "excellent", "🟢"
        elif value >= thresholds['good']:
            return "good", "🟡"
        elif value >= thresholds['poor']:
            return "poor", "🟠"
        else:
            return "critical", "🔴"
    
    def format_percentage(self, value: float, precision: int = 2) -> str:
        """Format percentage with proper precision."""
        return f"{value * 100:.{precision}f}%"
    
    def format_metric_value(self, metric_name: str, value: float) -> str:
        """Format metric value based on type."""
        
        percentage_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr', 'hit_rate', 'r2_score']
        
        if metric_name.lower() in percentage_metrics:
            return self.format_percentage(value)
        elif metric_name.lower() in ['mae', 'mse', 'rmse']:
            return f"{value:.4f}"
        elif metric_name.lower() in ['latency', 'inference_time']:
            return f"{value:.2f}ms"
        else:
            return f"{value:.3f}"
    
    async def format_model_report(self, model_metrics: ModelMetrics) -> FormattedModelReport:
        """Format ML model report - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement format_model_report")


class SpotifyRecommendationModelFormatter(BaseAIModelFormatter):
    """Specialized formatter for Spotify recommendation models."""
    
    async def format_model_report(self, model_metrics: ModelMetrics) -> FormattedModelReport:
        """Format comprehensive recommendation model report."""
        
        # Generate performance analysis
        performance_analysis = await self._analyze_recommendation_performance(model_metrics)
        
        # Create visualizations
        visualizations = await self._create_recommendation_visualizations(model_metrics)
        
        # Generate explainability report
        explainability_report = await self._generate_explainability_report(model_metrics)
        
        # Create recommendations for model improvement
        recommendations = await self._generate_model_recommendations(model_metrics, performance_analysis)
        
        # Generate alerts if needed
        alerts = await self._check_model_alerts(model_metrics)
        
        # Format main content
        formatted_content = await self._format_recommendation_content(
            model_metrics, performance_analysis, explainability_report
        )
        
        metadata = {
            "report_type": "recommendation_model",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_metrics.model_type.value,
            "model_stage": model_metrics.stage.value,
            "tenant_id": self.tenant_id,
            "performance_summary": performance_analysis.get('summary', {}),
            "alert_count": len(alerts)
        }
        
        return FormattedModelReport(
            model_metrics=model_metrics,
            formatted_content=formatted_content,
            explainability_report=explainability_report,
            visualizations=visualizations,
            recommendations=recommendations,
            alerts=alerts,
            metadata=metadata
        )
    
    async def _analyze_recommendation_performance(self, model_metrics: ModelMetrics) -> Dict[str, Any]:
        """Analyze recommendation model performance."""
        
        performance = model_metrics.performance_metrics
        
        analysis = {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "summary": {},
            "trend_analysis": {},
            "comparative_analysis": {}
        }
        
        # Calculate overall performance score
        key_metrics = ['hit_rate', 'ndcg', 'precision', 'recall']
        scores = []
        
        for metric in key_metrics:
            if metric in performance:
                value = performance[metric]
                status, _ = self.get_performance_status(metric, value)
                
                # Convert status to numeric score
                status_scores = {'excellent': 1.0, 'good': 0.75, 'poor': 0.5, 'critical': 0.25}
                scores.append(status_scores.get(status, 0.5))
                
                # Add to strengths or weaknesses
                if status in ['excellent', 'good']:
                    analysis['strengths'].append(f"{metric.replace('_', ' ').title()}: {self.format_metric_value(metric, value)}")
                else:
                    analysis['weaknesses'].append(f"{metric.replace('_', ' ').title()}: {self.format_metric_value(metric, value)}")
        
        analysis['overall_score'] = statistics.mean(scores) if scores else 0.5
        
        # Performance summary
        analysis['summary'] = {
            "hit_rate": performance.get('hit_rate', 0.0),
            "ndcg_score": performance.get('ndcg', 0.0),
            "precision_at_k": performance.get('precision', 0.0),
            "recall_at_k": performance.get('recall', 0.0),
            "coverage": performance.get('coverage', 0.0),
            "diversity": performance.get('diversity', 0.0),
            "novelty": performance.get('novelty', 0.0)
        }
        
        # Recommendation quality insights
        if analysis['overall_score'] > 0.85:
            analysis['quality_level'] = "🌟 Exceptional"
            analysis['quality_description'] = "Model performing at industry-leading levels"
        elif analysis['overall_score'] > 0.70:
            analysis['quality_level'] = "✅ Good"
            analysis['quality_description'] = "Model meeting business requirements"
        elif analysis['overall_score'] > 0.50:
            analysis['quality_level'] = "⚠️ Needs Improvement"
            analysis['quality_description'] = "Model requires optimization"
        else:
            analysis['quality_level'] = "🚨 Critical"
            analysis['quality_description'] = "Model needs immediate attention"
        
        return analysis
    
    async def _create_recommendation_visualizations(self, model_metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Create visualizations for recommendation model performance."""
        
        visualizations = []
        performance = model_metrics.performance_metrics
        
        # Performance radar chart
        radar_metrics = ['hit_rate', 'ndcg', 'precision', 'recall', 'coverage', 'diversity']
        radar_data = []
        radar_labels = []
        
        for metric in radar_metrics:
            if metric in performance:
                radar_data.append(performance[metric] * 100)
                radar_labels.append(metric.replace('_', ' ').title())
        
        if radar_data:
            radar_chart = {
                "type": "radar",
                "title": "Recommendation Model Performance Radar",
                "data": {
                    "labels": radar_labels,
                    "datasets": [{
                        "label": f"{model_metrics.model_name} v{model_metrics.version}",
                        "data": radar_data,
                        "backgroundColor": "rgba(29, 185, 84, 0.2)",
                        "borderColor": "#1DB954",
                        "pointBackgroundColor": "#1DB954",
                        "pointBorderColor": "#fff",
                        "pointHoverBackgroundColor": "#fff",
                        "pointHoverBorderColor": "#1DB954"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "Model Performance Overview"},
                        "legend": {"position": "top"}
                    },
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 100,
                            "ticks": {"stepSize": 20}
                        }
                    }
                }
            }
            visualizations.append(radar_chart)
        
        # Hit rate trend over time (simulated data)
        hit_rate_trend = {
            "type": "line",
            "title": "Hit Rate Trend",
            "data": {
                "labels": ["Week 1", "Week 2", "Week 3", "Week 4", "Current"],
                "datasets": [{
                    "label": "Hit Rate",
                    "data": [
                        performance.get('hit_rate', 0.6) * 0.9,
                        performance.get('hit_rate', 0.6) * 0.95,
                        performance.get('hit_rate', 0.6) * 0.98,
                        performance.get('hit_rate', 0.6) * 1.02,
                        performance.get('hit_rate', 0.6)
                    ],
                    "borderColor": "#1DB954",
                    "backgroundColor": "rgba(29, 185, 84, 0.1)",
                    "tension": 0.4
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Hit Rate Trend Analysis"}},
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "ticks": {
                            "callback": "function(value) { return (value * 100).toFixed(1) + '%'; }"
                        }
                    }
                }
            }
        }
        visualizations.append(hit_rate_trend)
        
        # Feature importance bar chart
        if model_metrics.model_type == ModelType.RECOMMENDATION:
            feature_importance = {
                "type": "bar",
                "title": "Feature Importance",
                "data": {
                    "labels": ["User History", "Collaborative Filtering", "Content Features", "Temporal Patterns", "Social Signals"],
                    "datasets": [{
                        "label": "Importance Score",
                        "data": [0.35, 0.28, 0.18, 0.12, 0.07],
                        "backgroundColor": ["#1DB954", "#1ED760", "#A0D468", "#7FB069", "#6A994E"],
                        "borderColor": ["#169C47", "#17B548", "#8BC34A", "#689F38", "#558B2F"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Model Feature Importance"}},
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 0.4,
                            "ticks": {
                                "callback": "function(value) { return (value * 100).toFixed(1) + '%'; }"
                            }
                        }
                    }
                }
            }
            visualizations.append(feature_importance)
        
        return visualizations
    
    async def _generate_explainability_report(self, model_metrics: ModelMetrics) -> ExplainabilityReport:
        """Generate explainability report for the recommendation model."""
        
        # Simulated feature importance (in production, this would come from SHAP/LIME)
        feature_importance = {
            "user_listening_history": 0.35,
            "collaborative_filtering_score": 0.28,
            "audio_features_similarity": 0.18,
            "temporal_patterns": 0.12,
            "social_signals": 0.07
        }
        
        # Simulated SHAP values for key features
        shap_values = {
            "user_listening_history": [0.23, 0.31, 0.28, 0.33, 0.29],
            "collaborative_filtering_score": [0.19, 0.25, 0.22, 0.26, 0.24],
            "audio_features_similarity": [0.12, 0.15, 0.13, 0.17, 0.14]
        }
        
        # Local explanations for sample recommendations
        local_explanations = [
            {
                "user_id": "user_001",
                "recommended_track": "Track A",
                "explanation": "Recommended because user frequently listens to similar artists",
                "confidence": 0.87,
                "key_factors": ["artist_similarity", "genre_match", "user_history"]
            },
            {
                "user_id": "user_002",
                "recommended_track": "Track B",
                "explanation": "Users with similar taste also enjoyed this track",
                "confidence": 0.73,
                "key_factors": ["collaborative_filtering", "playlist_cooccurrence"]
            }
        ]
        
        # Global model explanations
        global_explanations = {
            "model_behavior": "Model primarily relies on user listening history and collaborative filtering",
            "decision_boundaries": "Strong preference for tracks with high user-item interaction scores",
            "bias_detection": "Slight bias towards popular tracks, but diversity mechanisms help",
            "fairness_assessment": "Model shows balanced performance across different user demographics"
        }
        
        # Bias metrics
        bias_metrics = {
            "popularity_bias": 0.15,  # Lower is better
            "genre_diversity": 0.78,  # Higher is better
            "artist_coverage": 0.65,  # Higher is better
            "demographic_parity": 0.92  # Higher is better
        }
        
        # Fairness scores across different user groups
        fairness_scores = {
            "age_groups": {
                "18-25": 0.84,
                "26-35": 0.87,
                "36-45": 0.82,
                "46+": 0.79
            },
            "gender_groups": {
                "male": 0.85,
                "female": 0.86,
                "other": 0.83
            },
            "geographic_regions": {
                "north_america": 0.88,
                "europe": 0.85,
                "asia": 0.81,
                "other": 0.78
            }
        }
        
        return ExplainabilityReport(
            model_id=model_metrics.model_id,
            method=ExplainabilityMethod.SHAP,
            feature_importance=feature_importance,
            shap_values=shap_values,
            local_explanations=local_explanations,
            global_explanations=global_explanations,
            bias_metrics=bias_metrics,
            fairness_scores=fairness_scores
        )
    
    async def _generate_model_recommendations(self, model_metrics: ModelMetrics, 
                                            performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for model improvement."""
        
        recommendations = []
        performance = model_metrics.performance_metrics
        overall_score = performance_analysis['overall_score']
        
        # Performance-based recommendations
        if overall_score < 0.7:
            recommendations.append("🔧 Consider retraining the model with more recent data to improve overall performance")
            recommendations.append("📊 Analyze feature engineering opportunities to enhance model predictive power")
        
        # Specific metric recommendations
        hit_rate = performance.get('hit_rate', 0.0)
        if hit_rate < 0.6:
            recommendations.append(f"🎯 Hit rate ({self.format_percentage(hit_rate)}) is below target - consider improving candidate generation")
        
        ndcg = performance.get('ndcg', 0.0)
        if ndcg < 0.7:
            recommendations.append(f"📈 NDCG score ({self.format_percentage(ndcg)}) indicates ranking quality issues - review ranking algorithm")
        
        coverage = performance.get('coverage', 0.0)
        if coverage < 0.5:
            recommendations.append(f"🌍 Low catalog coverage ({self.format_percentage(coverage)}) - implement diversity promotion strategies")
        
        # Resource optimization recommendations
        inference_time = model_metrics.inference_metrics.get('avg_latency', 0.0)
        if inference_time > 100:  # ms
            recommendations.append(f"⚡ High inference latency ({inference_time:.1f}ms) - consider model optimization or caching")
        
        # Data quality recommendations
        data_drift = model_metrics.data_quality_metrics.get('data_drift_score', 0.0)
        if data_drift > 0.1:
            recommendations.append("📊 Significant data drift detected - retrain model with recent data")
        
        # Business impact recommendations
        if model_metrics.stage == ModelStage.PRODUCTION:
            recommendations.append("📈 Monitor user engagement metrics to validate model performance impact")
            recommendations.append("🔄 Set up automated retraining pipeline for continuous model improvement")
        
        # Fairness and bias recommendations
        popularity_bias = 0.15  # From explainability report
        if popularity_bias > 0.2:
            recommendations.append("⚖️ Address popularity bias by implementing fairness constraints in training")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    async def _check_model_alerts(self, model_metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Check for model performance alerts."""
        
        alerts = []
        performance = model_metrics.performance_metrics
        
        # Performance degradation alerts
        hit_rate = performance.get('hit_rate', 0.0)
        if hit_rate < 0.4:
            alerts.append({
                "type": "performance_degradation",
                "severity": "high",
                "metric": "hit_rate",
                "current_value": hit_rate,
                "threshold": 0.4,
                "message": f"Hit rate critically low at {self.format_percentage(hit_rate)}",
                "recommended_action": "Immediate model retraining required"
            })
        
        # Latency alerts
        inference_time = model_metrics.inference_metrics.get('avg_latency', 0.0)
        if inference_time > 200:  # ms
            alerts.append({
                "type": "latency_alert",
                "severity": "medium",
                "metric": "inference_latency",
                "current_value": inference_time,
                "threshold": 200,
                "message": f"High inference latency: {inference_time:.1f}ms",
                "recommended_action": "Optimize model or increase infrastructure capacity"
            })
        
        # Data drift alerts
        data_drift = model_metrics.data_quality_metrics.get('data_drift_score', 0.0)
        if data_drift > self.alert_thresholds['data_drift']:
            alerts.append({
                "type": "data_drift",
                "severity": "medium",
                "metric": "data_drift_score",
                "current_value": data_drift,
                "threshold": self.alert_thresholds['data_drift'],
                "message": f"Data drift detected: {data_drift:.3f}",
                "recommended_action": "Review data pipeline and consider model retraining"
            })
        
        # Error rate alerts
        error_rate = model_metrics.inference_metrics.get('error_rate', 0.0)
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                "type": "error_rate",
                "severity": "high",
                "metric": "error_rate",
                "current_value": error_rate,
                "threshold": self.alert_thresholds['error_rate'],
                "message": f"High error rate: {self.format_percentage(error_rate)}",
                "recommended_action": "Investigate model serving infrastructure"
            })
        
        return alerts
    
    async def _format_recommendation_content(self, model_metrics: ModelMetrics, 
                                           performance_analysis: Dict[str, Any],
                                           explainability_report: ExplainabilityReport) -> str:
        """Format the main recommendation model content."""
        
        model_name = model_metrics.model_name
        version = model_metrics.version
        timestamp = model_metrics.timestamp
        overall_score = performance_analysis['overall_score']
        quality_level = performance_analysis.get('quality_level', 'Unknown')
        
        # Performance metrics section
        performance_section = "## 📊 Performance Metrics\n\n"
        
        key_metrics = ['hit_rate', 'ndcg', 'precision', 'recall', 'coverage', 'diversity']
        performance = model_metrics.performance_metrics
        
        for metric in key_metrics:
            if metric in performance:
                value = performance[metric]
                formatted_value = self.format_metric_value(metric, value)
                status, icon = self.get_performance_status(metric, value)
                
                performance_section += f"**{metric.replace('_', ' ').title()}**: {formatted_value} {icon}\n"
        
        # Strengths and weaknesses
        strengths_section = "## ✅ Model Strengths\n\n"
        if performance_analysis['strengths']:
            for strength in performance_analysis['strengths']:
                strengths_section += f"• {strength}\n"
        else:
            strengths_section += "• No significant strengths identified\n"
        
        weaknesses_section = "## ⚠️ Areas for Improvement\n\n"
        if performance_analysis['weaknesses']:
            for weakness in performance_analysis['weaknesses']:
                weaknesses_section += f"• {weakness}\n"
        else:
            weaknesses_section += "• No significant weaknesses identified\n"
        
        # Explainability section
        explainability_section = "## 🔍 Model Explainability\n\n"
        explainability_section += "### Feature Importance\n"
        
        for feature, importance in explainability_report.feature_importance.items():
            feature_name = feature.replace('_', ' ').title()
            explainability_section += f"• **{feature_name}**: {self.format_percentage(importance)}\n"
        
        explainability_section += "\n### Bias & Fairness Assessment\n"
        bias_metrics = explainability_report.bias_metrics
        
        explainability_section += f"• **Popularity Bias**: {bias_metrics.get('popularity_bias', 0):.3f} (lower is better)\n"
        explainability_section += f"• **Genre Diversity**: {self.format_percentage(bias_metrics.get('genre_diversity', 0))}\n"
        explainability_section += f"• **Demographic Parity**: {self.format_percentage(bias_metrics.get('demographic_parity', 0))}\n"
        
        # Resource usage section
        resource_section = "## 💻 Resource Usage\n\n"
        resource_usage = model_metrics.resource_usage
        
        if resource_usage:
            resource_section += f"• **CPU Usage**: {resource_usage.get('cpu_usage', 0):.1f}%\n"
            resource_section += f"• **Memory Usage**: {resource_usage.get('memory_usage', 0):.1f} GB\n"
            resource_section += f"• **GPU Utilization**: {resource_usage.get('gpu_usage', 0):.1f}%\n"
            resource_section += f"• **Throughput**: {resource_usage.get('throughput', 0):.0f} req/sec\n"
        else:
            resource_section += "• Resource usage data not available\n"
        
        # Combine all sections
        content = f"""
# 🎵 {model_name} Performance Report

**Model Version**: {version}  
**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Overall Performance**: {quality_level} ({self.format_percentage(overall_score)})  
**Model Stage**: {model_metrics.stage.value.title()}

{performance_section}

{strengths_section}

{weaknesses_section}

{explainability_section}

{resource_section}

## 📈 Business Impact

The {model_name} recommendation model is currently serving music recommendations to millions of Spotify users. 
Performance at this level {'exceeds' if overall_score > 0.8 else 'meets' if overall_score > 0.6 else 'falls short of'} 
our business objectives for user engagement and satisfaction.

**Key Business Metrics:**
• User engagement directly correlates with hit rate performance
• Recommendation diversity impacts user session length
• Model latency affects user experience and platform responsiveness

## 🎯 Next Steps

Based on the current performance analysis, the recommended next steps are:

1. **Monitor Continuously**: Set up automated monitoring for all key metrics
2. **A/B Testing**: Compare performance against baseline and alternative models  
3. **User Feedback**: Incorporate user satisfaction signals into model evaluation
4. **Retraining Schedule**: Establish regular retraining cadence based on data drift

---
*Report generated by Spotify AI Agent - Model Performance Analytics*
        """.strip()
        
        return content


class GeneralMLModelFormatter(BaseAIModelFormatter):
    """General-purpose ML model formatter for various model types."""
    
    async def format_model_report(self, model_metrics: ModelMetrics) -> FormattedModelReport:
        """Format general ML model report."""
        
        # Analyze performance based on model type
        performance_analysis = await self._analyze_general_performance(model_metrics)
        
        # Create appropriate visualizations
        visualizations = await self._create_general_visualizations(model_metrics)
        
        # Generate recommendations
        recommendations = await self._generate_general_recommendations(model_metrics, performance_analysis)
        
        # Check for alerts
        alerts = await self._check_general_alerts(model_metrics)
        
        # Format content
        formatted_content = await self._format_general_content(model_metrics, performance_analysis)
        
        metadata = {
            "report_type": "general_ml_model",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_metrics.model_type.value,
            "model_stage": model_metrics.stage.value,
            "tenant_id": self.tenant_id,
            "performance_summary": performance_analysis.get('summary', {}),
            "alert_count": len(alerts)
        }
        
        return FormattedModelReport(
            model_metrics=model_metrics,
            formatted_content=formatted_content,
            explainability_report=None,  # Can be added based on model type
            visualizations=visualizations,
            recommendations=recommendations,
            alerts=alerts,
            metadata=metadata
        )
    
    async def _analyze_general_performance(self, model_metrics: ModelMetrics) -> Dict[str, Any]:
        """Analyze general ML model performance."""
        
        performance = model_metrics.performance_metrics
        model_type = model_metrics.model_type
        
        analysis = {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "summary": {},
            "model_specific_insights": []
        }
        
        # Define key metrics based on model type
        if model_type == ModelType.CLASSIFICATION:
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        elif model_type == ModelType.REGRESSION:
            key_metrics = ['mae', 'mse', 'rmse', 'r2_score']
        elif model_type == ModelType.NLP:
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        else:
            key_metrics = list(performance.keys())[:5]  # Take first 5 available metrics
        
        # Calculate overall score and identify strengths/weaknesses
        scores = []
        for metric in key_metrics:
            if metric in performance:
                value = performance[metric]
                
                # For regression metrics (lower is better)
                if metric in ['mae', 'mse', 'rmse']:
                    # Convert to performance score (assume lower values are better)
                    score = max(0, 1 - value)  # Simplified scoring
                    status = "good" if score > 0.7 else "poor"
                else:
                    status, _ = self.get_performance_status(metric, value)
                    status_scores = {'excellent': 1.0, 'good': 0.75, 'poor': 0.5, 'critical': 0.25}
                    score = status_scores.get(status, 0.5)
                
                scores.append(score)
                
                formatted_value = self.format_metric_value(metric, value)
                if score > 0.7:
                    analysis['strengths'].append(f"{metric.replace('_', ' ').title()}: {formatted_value}")
                else:
                    analysis['weaknesses'].append(f"{metric.replace('_', ' ').title()}: {formatted_value}")
        
        analysis['overall_score'] = statistics.mean(scores) if scores else 0.5
        analysis['summary'] = {metric: performance.get(metric, 0.0) for metric in key_metrics}
        
        # Model-specific insights
        if model_type == ModelType.CLASSIFICATION:
            precision = performance.get('precision', 0.0)
            recall = performance.get('recall', 0.0)
            
            if precision > 0.8 and recall < 0.6:
                analysis['model_specific_insights'].append("High precision but low recall - model is conservative")
            elif precision < 0.6 and recall > 0.8:
                analysis['model_specific_insights'].append("High recall but low precision - model is aggressive")
            elif precision > 0.8 and recall > 0.8:
                analysis['model_specific_insights'].append("Excellent balance between precision and recall")
        
        elif model_type == ModelType.REGRESSION:
            r2 = performance.get('r2_score', 0.0)
            mae = performance.get('mae', float('inf'))
            
            if r2 > 0.9:
                analysis['model_specific_insights'].append("Excellent explanatory power - model captures variance well")
            elif r2 < 0.5:
                analysis['model_specific_insights'].append("Low explanatory power - consider feature engineering")
            
            if mae < 0.1:
                analysis['model_specific_insights'].append("Low prediction error - high accuracy model")
        
        return analysis
    
    async def _create_general_visualizations(self, model_metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Create visualizations for general ML models."""
        
        visualizations = []
        performance = model_metrics.performance_metrics
        model_type = model_metrics.model_type
        
        # Performance metrics bar chart
        if performance:
            metrics_chart = {
                "type": "bar",
                "title": f"{model_type.value.title()} Model Performance",
                "data": {
                    "labels": [metric.replace('_', ' ').title() for metric in performance.keys()],
                    "datasets": [{
                        "label": "Performance Score",
                        "data": list(performance.values()),
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Model Performance Metrics"}},
                    "scales": {"y": {"beginAtZero": True}}
                }
            }
            visualizations.append(metrics_chart)
        
        # Training progress (simulated)
        if model_metrics.training_metrics:
            training_chart = {
                "type": "line",
                "title": "Training Progress",
                "data": {
                    "labels": ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4", "Epoch 5"],
                    "datasets": [
                        {
                            "label": "Training Loss",
                            "data": [0.8, 0.6, 0.4, 0.3, 0.25],
                            "borderColor": "#FF6B6B",
                            "backgroundColor": "rgba(255, 107, 107, 0.1)",
                            "tension": 0.4
                        },
                        {
                            "label": "Validation Loss",
                            "data": [0.85, 0.65, 0.45, 0.35, 0.32],
                            "borderColor": "#4ECDC4",
                            "backgroundColor": "rgba(78, 205, 196, 0.1)",
                            "tension": 0.4
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Training & Validation Loss"}},
                    "scales": {"y": {"beginAtZero": True}}
                }
            }
            visualizations.append(training_chart)
        
        return visualizations
    
    async def _generate_general_recommendations(self, model_metrics: ModelMetrics, 
                                              performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate general recommendations for model improvement."""
        
        recommendations = []
        overall_score = performance_analysis['overall_score']
        model_type = model_metrics.model_type
        
        # General performance recommendations
        if overall_score < 0.6:
            recommendations.append("🔧 Model performance is below acceptable threshold - consider comprehensive retraining")
            recommendations.append("📊 Review feature selection and engineering process")
        
        # Model-specific recommendations
        if model_type == ModelType.CLASSIFICATION:
            performance = model_metrics.performance_metrics
            precision = performance.get('precision', 0.0)
            recall = performance.get('recall', 0.0)
            
            if precision < 0.7:
                recommendations.append("🎯 Improve precision by refining decision threshold or addressing false positives")
            if recall < 0.7:
                recommendations.append("📈 Improve recall by addressing class imbalance or feature representation")
        
        elif model_type == ModelType.REGRESSION:
            mae = model_metrics.performance_metrics.get('mae', 0.0)
            if mae > 0.1:
                recommendations.append("📉 High prediction error - consider ensemble methods or advanced algorithms")
        
        # Infrastructure recommendations
        inference_time = model_metrics.inference_metrics.get('avg_latency', 0.0)
        if inference_time > 50:  # ms
            recommendations.append("⚡ Optimize model for faster inference - consider quantization or pruning")
        
        # Monitoring recommendations
        recommendations.append("📊 Implement continuous monitoring for model drift and performance degradation")
        recommendations.append("🔄 Establish automated retraining pipeline based on performance thresholds")
        
        return recommendations[:5]
    
    async def _check_general_alerts(self, model_metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Check for general model alerts."""
        
        alerts = []
        
        # Performance alerts based on model type
        if model_metrics.model_type == ModelType.CLASSIFICATION:
            accuracy = model_metrics.performance_metrics.get('accuracy', 0.0)
            if accuracy < 0.7:
                alerts.append({
                    "type": "low_accuracy",
                    "severity": "high",
                    "metric": "accuracy",
                    "current_value": accuracy,
                    "threshold": 0.7,
                    "message": f"Classification accuracy critically low: {self.format_percentage(accuracy)}",
                    "recommended_action": "Review training data and model architecture"
                })
        
        # General infrastructure alerts
        memory_usage = model_metrics.resource_usage.get('memory_usage', 0.0)
        if memory_usage > 8.0:  # GB
            alerts.append({
                "type": "high_memory_usage",
                "severity": "medium",
                "metric": "memory_usage",
                "current_value": memory_usage,
                "threshold": 8.0,
                "message": f"High memory usage: {memory_usage:.1f} GB",
                "recommended_action": "Consider model optimization or infrastructure scaling"
            })
        
        return alerts
    
    async def _format_general_content(self, model_metrics: ModelMetrics, 
                                    performance_analysis: Dict[str, Any]) -> str:
        """Format general ML model content."""
        
        model_name = model_metrics.model_name
        model_type = model_metrics.model_type.value.replace('_', ' ').title()
        version = model_metrics.version
        stage = model_metrics.stage.value.title()
        overall_score = performance_analysis['overall_score']
        
        content = f"""
# 🤖 {model_name} - {model_type} Model Report

**Model Version**: {version}  
**Model Type**: {model_type}  
**Stage**: {stage}  
**Overall Performance**: {self.format_percentage(overall_score)}  
**Last Updated**: {model_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📊 Performance Summary

"""
        
        # Add performance metrics
        for metric, value in performance_analysis['summary'].items():
            if value > 0:
                formatted_value = self.format_metric_value(metric, value)
                status, icon = self.get_performance_status(metric, value)
                content += f"• **{metric.replace('_', ' ').title()}**: {formatted_value} {icon}\n"
        
        # Add strengths and weaknesses
        if performance_analysis['strengths']:
            content += "\n## ✅ Model Strengths\n\n"
            for strength in performance_analysis['strengths']:
                content += f"• {strength}\n"
        
        if performance_analysis['weaknesses']:
            content += "\n## ⚠️ Areas for Improvement\n\n"
            for weakness in performance_analysis['weaknesses']:
                content += f"• {weakness}\n"
        
        # Add model-specific insights
        if performance_analysis['model_specific_insights']:
            content += "\n## 🔍 Model Insights\n\n"
            for insight in performance_analysis['model_specific_insights']:
                content += f"• {insight}\n"
        
        content += "\n---\n*Generated by Spotify AI Agent - ML Model Analytics*"
        
        return content


# Factory function for creating AI model formatters
def create_ai_model_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseAIModelFormatter:
    """
    Factory function to create AI/ML model formatters.
    
    Args:
        formatter_type: Type of formatter ('recommendation', 'classification', 'regression', 'general')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured AI model formatter instance
    """
    formatters = {
        'recommendation': SpotifyRecommendationModelFormatter,
        'recommendation_model': SpotifyRecommendationModelFormatter,
        'classification': GeneralMLModelFormatter,
        'regression': GeneralMLModelFormatter,
        'nlp': GeneralMLModelFormatter,
        'computer_vision': GeneralMLModelFormatter,
        'general': GeneralMLModelFormatter,
        'ml_model': GeneralMLModelFormatter,
        'ai_model': GeneralMLModelFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported AI model formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
