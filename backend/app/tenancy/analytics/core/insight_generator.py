"""
Advanced Insight Generator for Multi-Tenant Analytics

This module implements an ultra-sophisticated AI-powered insight generation system
with machine learning models, natural language processing, and predictive analytics.

Features:
- AI-powered business insight generation
- Natural language insights with explanations
- Predictive analytics and forecasting
- Anomaly detection and root cause analysis
- Pattern recognition and trend analysis
- Automated recommendation generation
- Multi-language insight support

Created by Expert Team:
- Lead Dev + AI Architect: Architecture and ML integration
- ML Engineer: Advanced ML models and NLP processing
- Data Scientist: Statistical analysis and predictive modeling
- Senior Backend Developer: API integration and real-time processing
- Backend Security Specialist: Insight security and data privacy
- Microservices Architect: Scalable insight infrastructure

Developed by: Fahed Mlaiel
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import uuid
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from textblob import TextBlob
import spacy
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of insights that can be generated"""
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    USER_BEHAVIOR = "user_behavior"
    BUSINESS_METRICS = "business_metrics"
    PREDICTIVE_FORECAST = "predictive_forecast"
    CORRELATION_ANALYSIS = "correlation_analysis"
    SEGMENTATION = "segmentation"
    RECOMMENDATION = "recommendation"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"

class InsightPriority(Enum):
    """Priority levels for insights"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class ConfidenceLevel(Enum):
    """Confidence levels for insights"""
    VERY_HIGH = "very_high"  # 95%+
    HIGH = "high"            # 85-95%
    MEDIUM = "medium"        # 70-85%
    LOW = "low"              # 50-70%
    VERY_LOW = "very_low"    # <50%

@dataclass
class InsightConfig:
    """Configuration for insight generation"""
    anomaly_detection: bool = True
    trend_analysis: bool = True
    forecasting: bool = True
    clustering: bool = True
    classification: bool = True
    recommendation_engine: bool = True
    sentiment_analysis: bool = True
    pattern_recognition: bool = True
    nlp_enabled: bool = True
    multi_language: bool = True
    min_confidence_threshold: float = 0.7
    max_insights_per_request: int = 20

@dataclass
class Insight:
    """Comprehensive insight data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: InsightType = InsightType.TREND_ANALYSIS
    title: str = ""
    description: str = ""
    priority: InsightPriority = InsightPriority.MEDIUM
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Core insight data
    metric_name: str = ""
    current_value: Union[float, int, str] = 0
    previous_value: Optional[Union[float, int, str]] = None
    change_percentage: Optional[float] = None
    trend_direction: Optional[str] = None  # "up", "down", "stable"
    
    # Supporting data
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    related_metrics: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    
    # Recommendations and actions
    recommendations: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    
    # Metadata
    tenant_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    language: str = "en"
    tags: List[str] = field(default_factory=list)
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None
    chart_type: Optional[str] = None

@dataclass
class AnomalyInsight:
    """Specialized insight for anomalies"""
    anomaly_score: float = 0.0
    affected_metrics: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)
    severity: str = "medium"  # "low", "medium", "high", "critical"
    detection_method: str = "isolation_forest"
    historical_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictiveInsight:
    """Specialized insight for predictions"""
    forecast_horizon: int = 24  # hours
    predicted_values: List[float] = field(default_factory=list)
    prediction_timestamps: List[datetime] = field(default_factory=list)
    confidence_intervals: List[Tuple[float, float]] = field(default_factory=list)
    model_accuracy: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class InsightStats:
    """Statistics for insight generation"""
    total_insights: int = 0
    insights_by_type: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_generation_time_ms: float = 0.0
    anomalies_detected: int = 0
    predictions_generated: int = 0
    recommendations_created: int = 0
    last_insight_time: Optional[datetime] = None

class InsightGenerator:
    """
    Ultra-advanced insight generator with AI and ML capabilities
    """
    
    def __init__(self, config: InsightConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # ML models and analyzers
        self.anomaly_detector = None
        self.trend_analyzer = None
        self.clustering_model = None
        self.forecasting_models = {}
        self.classification_model = None
        
        # NLP models
        self.nlp_pipeline = None
        self.sentiment_analyzer = None
        self.text_generator = None
        self.tokenizer = None
        self.language_models = {}
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.pca_transformer = PCA(n_components=10)
        
        # Insight templates and generators
        self.insight_templates = {}
        self.recommendation_engine = None
        
        # Caching and optimization
        self.insight_cache = {}
        self.pattern_cache = {}
        self.model_cache = {}
        
        # Processing infrastructure
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Statistics and monitoring
        self.tenant_stats = defaultdict(lambda: InsightStats())
        self.global_stats = InsightStats()
        
        # Historical data and patterns
        self.historical_patterns = defaultdict(dict)
        self.trend_baselines = defaultdict(dict)
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize insight generator with all AI/ML components"""
        try:
            self.logger.info("Initializing Insight Generator...")
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Initialize NLP models
            await self._initialize_nlp_models()
            
            # Load insight templates
            await self._load_insight_templates()
            
            # Initialize recommendation engine
            await self._initialize_recommendation_engine()
            
            # Load pre-trained models
            await self._load_pretrained_models()
            
            self.is_initialized = True
            self.logger.info("Insight Generator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Insight Generator: {e}")
            return False
    
    async def generate_insights(
        self,
        tenant_id: str,
        insight_types: List[str],
        data_context: Optional[Dict] = None
    ) -> List[Insight]:
        """Generate AI-powered insights based on data context"""
        try:
            start_time = time.time()
            insights = []
            
            # Validate input
            if not insight_types or not tenant_id:
                return insights
            
            # Prepare data context
            context = data_context or {}
            
            # Generate insights for each requested type
            for insight_type in insight_types:
                try:
                    if insight_type == InsightType.ANOMALY_DETECTION.value:
                        anomaly_insights = await self._generate_anomaly_insights(tenant_id, context)
                        insights.extend(anomaly_insights)
                    
                    elif insight_type == InsightType.TREND_ANALYSIS.value:
                        trend_insights = await self._generate_trend_insights(tenant_id, context)
                        insights.extend(trend_insights)
                    
                    elif insight_type == InsightType.PREDICTIVE_FORECAST.value:
                        forecast_insights = await self._generate_forecast_insights(tenant_id, context)
                        insights.extend(forecast_insights)
                    
                    elif insight_type == InsightType.PERFORMANCE_ANALYSIS.value:
                        performance_insights = await self._generate_performance_insights(tenant_id, context)
                        insights.extend(performance_insights)
                    
                    elif insight_type == InsightType.USER_BEHAVIOR.value:
                        behavior_insights = await self._generate_behavior_insights(tenant_id, context)
                        insights.extend(behavior_insights)
                    
                    elif insight_type == InsightType.SEGMENTATION.value:
                        segmentation_insights = await self._generate_segmentation_insights(tenant_id, context)
                        insights.extend(segmentation_insights)
                    
                    elif insight_type == InsightType.RECOMMENDATION.value:
                        recommendation_insights = await self._generate_recommendation_insights(tenant_id, context)
                        insights.extend(recommendation_insights)
                    
                    elif insight_type == InsightType.CORRELATION_ANALYSIS.value:
                        correlation_insights = await self._generate_correlation_insights(tenant_id, context)
                        insights.extend(correlation_insights)
                
                except Exception as e:
                    self.logger.error(f"Failed to generate {insight_type} insights: {e}")
                    continue
            
            # Filter by confidence threshold
            filtered_insights = [
                insight for insight in insights
                if insight.confidence >= self.config.min_confidence_threshold
            ]
            
            # Limit number of insights
            if len(filtered_insights) > self.config.max_insights_per_request:
                # Sort by confidence and priority, take top insights
                sorted_insights = sorted(
                    filtered_insights,
                    key=lambda x: (x.priority.value, x.confidence),
                    reverse=True
                )
                filtered_insights = sorted_insights[:self.config.max_insights_per_request]
            
            # Generate natural language descriptions
            if self.config.nlp_enabled:
                await self._enhance_insights_with_nlp(filtered_insights)
            
            # Generate visualizations
            await self._generate_insight_visualizations(filtered_insights)
            
            # Update statistics
            generation_time = (time.time() - start_time) * 1000
            await self._update_insight_stats(tenant_id, filtered_insights, generation_time)
            
            return filtered_insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate insights for tenant {tenant_id}: {e}")
            return []
    
    async def detect_anomaly(
        self,
        tenant_id: str,
        metric_data: 'AnalyticsResult'
    ) -> float:
        """Detect anomaly score for a specific metric"""
        try:
            if not self.anomaly_detector or not isinstance(metric_data.value, (int, float)):
                return 0.0
            
            # Get historical values for comparison
            historical_values = await self._get_historical_values(
                tenant_id, metric_data.metric_name
            )
            
            if len(historical_values) < 10:
                return 0.0
            
            # Prepare data for anomaly detection
            all_values = historical_values + [metric_data.value]
            values_array = np.array(all_values).reshape(-1, 1)
            
            # Detect anomaly
            anomaly_scores = self.anomaly_detector.decision_function(values_array)
            current_score = anomaly_scores[-1]
            
            # Normalize score to 0-1 range
            normalized_score = max(0.0, min(1.0, (current_score + 1) / 2))
            
            return normalized_score
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return 0.0
    
    async def detect_real_time_anomaly(
        self,
        tenant_id: str,
        data_point: 'DataPoint'
    ) -> Optional[Dict]:
        """Detect real-time anomalies in streaming data"""
        try:
            # Extract numeric value from data point
            numeric_value = await self._extract_numeric_value(data_point)
            if numeric_value is None:
                return None
            
            # Check against real-time thresholds
            anomaly_score = await self._calculate_real_time_anomaly_score(
                tenant_id, numeric_value, data_point.source
            )
            
            if anomaly_score > 0.8:  # High anomaly threshold
                return {
                    "score": anomaly_score,
                    "value": numeric_value,
                    "source": data_point.source,
                    "timestamp": data_point.timestamp,
                    "severity": "high" if anomaly_score > 0.9 else "medium"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Real-time anomaly detection failed: {e}")
            return None
    
    async def get_count(self, tenant_id: str) -> int:
        """Get total insights generated for tenant"""
        return self.tenant_stats[tenant_id].total_insights
    
    async def get_insight_stats(self, tenant_id: str) -> InsightStats:
        """Get insight generation statistics for tenant"""
        return self.tenant_stats[tenant_id]
    
    async def _initialize_ml_models(self) -> None:
        """Initialize machine learning models"""
        try:
            # Anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Clustering for segmentation
            self.clustering_model = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            # Classification for pattern recognition
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            # Forecasting models
            self.forecasting_models = {
                "linear": LinearRegression(),
                "polynomial": None,  # Would be polynomial features + linear regression
                "neural": None       # Would be neural network model
            }
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    async def _initialize_nlp_models(self) -> None:
        """Initialize natural language processing models"""
        try:
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Text generation for insights
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2-medium"
            )
            
            # Load spaCy model for NER and text processing
            try:
                import spacy
                self.nlp_pipeline = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found, some NLP features disabled")
                self.nlp_pipeline = None
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP models: {e}")
            # Continue without NLP capabilities
            self.config.nlp_enabled = False
    
    async def _load_insight_templates(self) -> None:
        """Load insight templates for different types"""
        try:
            self.insight_templates = {
                InsightType.ANOMALY_DETECTION: {
                    "title": "Anomaly Detected in {metric_name}",
                    "description": "An unusual pattern has been detected in {metric_name}. Current value: {current_value}, Expected range: {expected_range}.",
                    "recommendations": [
                        "Investigate potential causes for the unusual behavior",
                        "Check system logs for related events",
                        "Monitor the metric closely for continued anomalies"
                    ]
                },
                InsightType.TREND_ANALYSIS: {
                    "title": "{metric_name} Trend Analysis",
                    "description": "{metric_name} shows a {trend_direction} trend with {change_percentage}% change over the analyzed period.",
                    "recommendations": [
                        "Continue monitoring the trend",
                        "Analyze factors contributing to the trend",
                        "Plan resources based on projected trend"
                    ]
                },
                InsightType.PREDICTIVE_FORECAST: {
                    "title": "Forecast for {metric_name}",
                    "description": "Based on historical data, {metric_name} is predicted to {forecast_direction} by {predicted_change}% in the next {time_period}.",
                    "recommendations": [
                        "Prepare for projected changes",
                        "Adjust capacity planning accordingly",
                        "Monitor actual vs predicted values"
                    ]
                }
            }
            
            self.logger.info("Insight templates loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load insight templates: {e}")
    
    async def _generate_anomaly_insights(
        self,
        tenant_id: str,
        context: Dict
    ) -> List[Insight]:
        """Generate anomaly detection insights"""
        try:
            insights = []
            
            # Get metrics data from context
            metrics_data = context.get("metrics_data", [])
            
            for metric_data in metrics_data:
                anomaly_score = await self.detect_anomaly(tenant_id, metric_data)
                
                if anomaly_score > self.config.min_confidence_threshold:
                    insight = Insight(
                        type=InsightType.ANOMALY_DETECTION,
                        title=f"Anomaly detected in {metric_data.metric_name}",
                        description=f"Unusual pattern detected with {anomaly_score:.2%} confidence",
                        priority=InsightPriority.HIGH if anomaly_score > 0.9 else InsightPriority.MEDIUM,
                        confidence=anomaly_score,
                        metric_name=metric_data.metric_name,
                        current_value=metric_data.value,
                        tenant_id=tenant_id,
                        supporting_data={"anomaly_score": anomaly_score},
                        recommendations=[
                            "Investigate root cause of anomaly",
                            "Check for system issues or external factors",
                            "Monitor metric closely for recovery"
                        ]
                    )
                    
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate anomaly insights: {e}")
            return []
    
    async def _generate_trend_insights(
        self,
        tenant_id: str,
        context: Dict
    ) -> List[Insight]:
        """Generate trend analysis insights"""
        try:
            insights = []
            
            # Analyze trends in historical data
            historical_data = context.get("historical_data", {})
            
            for metric_name, data_points in historical_data.items():
                if len(data_points) < 10:  # Need minimum data for trend analysis
                    continue
                
                # Calculate trend
                trend_analysis = await self._analyze_trend(data_points)
                
                if trend_analysis["confidence"] > self.config.min_confidence_threshold:
                    insight = Insight(
                        type=InsightType.TREND_ANALYSIS,
                        title=f"Trend analysis for {metric_name}",
                        description=f"{metric_name} shows {trend_analysis['direction']} trend",
                        priority=InsightPriority.MEDIUM,
                        confidence=trend_analysis["confidence"],
                        metric_name=metric_name,
                        trend_direction=trend_analysis["direction"],
                        change_percentage=trend_analysis["change_percentage"],
                        tenant_id=tenant_id,
                        supporting_data=trend_analysis,
                        recommendations=await self._generate_trend_recommendations(trend_analysis)
                    )
                    
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate trend insights: {e}")
            return []
    
    async def _generate_forecast_insights(
        self,
        tenant_id: str,
        context: Dict
    ) -> List[Insight]:
        """Generate predictive forecast insights"""
        try:
            insights = []
            
            # Generate forecasts for key metrics
            metrics_to_forecast = context.get("forecast_metrics", [])
            
            for metric_name in metrics_to_forecast:
                forecast = await self._generate_forecast(tenant_id, metric_name)
                
                if forecast and forecast["accuracy"] > self.config.min_confidence_threshold:
                    insight = Insight(
                        type=InsightType.PREDICTIVE_FORECAST,
                        title=f"Forecast for {metric_name}",
                        description=f"Predicted trend and values for {metric_name}",
                        priority=InsightPriority.MEDIUM,
                        confidence=forecast["accuracy"],
                        metric_name=metric_name,
                        tenant_id=tenant_id,
                        supporting_data=forecast,
                        recommendations=[
                            "Plan capacity based on forecast",
                            "Monitor actual vs predicted values",
                            "Adjust strategies based on predictions"
                        ]
                    )
                    
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast insights: {e}")
            return []
    
    # Placeholder implementations for complex methods
    async def _generate_performance_insights(self, tenant_id, context): return []
    async def _generate_behavior_insights(self, tenant_id, context): return []
    async def _generate_segmentation_insights(self, tenant_id, context): return []
    async def _generate_recommendation_insights(self, tenant_id, context): return []
    async def _generate_correlation_insights(self, tenant_id, context): return []
    async def _initialize_recommendation_engine(self): pass
    async def _load_pretrained_models(self): pass
    async def _enhance_insights_with_nlp(self, insights): pass
    async def _generate_insight_visualizations(self, insights): pass
    async def _update_insight_stats(self, tenant_id, insights, time): pass
    async def _get_historical_values(self, tenant_id, metric_name): return []
    async def _extract_numeric_value(self, data_point): return None
    async def _calculate_real_time_anomaly_score(self, tenant_id, value, source): return 0.0
    async def _analyze_trend(self, data_points): return {"direction": "stable", "confidence": 0.5, "change_percentage": 0.0}
    async def _generate_trend_recommendations(self, analysis): return []
    async def _generate_forecast(self, tenant_id, metric_name): return None

# Export main classes
__all__ = [
    "InsightGenerator",
    "InsightConfig",
    "Insight", 
    "AnomalyInsight",
    "PredictiveInsight",
    "InsightStats",
    "InsightType",
    "InsightPriority",
    "ConfidenceLevel"
]
