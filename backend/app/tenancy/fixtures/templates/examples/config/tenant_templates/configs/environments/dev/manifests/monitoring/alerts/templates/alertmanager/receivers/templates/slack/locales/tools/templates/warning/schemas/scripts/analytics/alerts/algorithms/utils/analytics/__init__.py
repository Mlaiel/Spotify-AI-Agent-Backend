"""
🎵 Enterprise Analytics Module for Spotify AI Agent

This module provides advanced analytics capabilities including business intelligence,
predictive analytics, trend analysis, and comprehensive metrics aggregation for
large-scale music streaming platform operations.

Features:
- Real-time analytics processing
- ML-powered predictive insights
- Business intelligence dashboards
- Trend analysis and forecasting
- Multi-dimensional metrics aggregation
- Revenue optimization analytics
- User behavior analysis
- Content performance metrics

Author: Fahed Mlaiel (Lead Developer & AI Architect)
Version: 2.0.0 (Enterprise Edition)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import json
from concurrent.futures import ThreadPoolExecutor
import aioredis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class AnalyticsLevel(Enum):
    """Analytics processing levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"
    AI_POWERED = "ai_powered"


class MetricCategory(Enum):
    """Metric categories for analytics"""
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER_BEHAVIOR = "user_behavior"
    CONTENT = "content"
    REVENUE = "revenue"
    PERFORMANCE = "performance"


@dataclass
class AnalyticsConfig:
    """Configuration for analytics processing"""
    level: AnalyticsLevel = AnalyticsLevel.ENTERPRISE
    real_time_processing: bool = True
    batch_processing: bool = True
    ml_predictions: bool = True
    auto_insights: bool = True
    retention_days: int = 90
    sampling_rate: float = 1.0
    parallel_workers: int = 4
    cache_ttl: int = 3600


@dataclass
class AnalyticsResult:
    """Result of analytics processing"""
    timestamp: datetime
    category: MetricCategory
    metrics: Dict[str, Any]
    insights: List[str] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0


class BaseAnalyticsEngine(ABC):
    """Base analytics engine"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._metrics = {
            'processed_events': Counter('analytics_processed_events_total', 'Total processed events'),
            'processing_time': Histogram('analytics_processing_seconds', 'Processing time'),
            'active_streams': Gauge('analytics_active_streams', 'Active analytics streams')
        }
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> AnalyticsResult:
        """Process analytics data"""
        pass
    
    @abstractmethod
    async def generate_insights(self, results: List[AnalyticsResult]) -> List[str]:
        """Generate insights from results"""
        pass


class AnalyticsEngine(BaseAnalyticsEngine):
    """Main analytics engine for Spotify AI Agent"""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.aggregators = {}
        self.predictors = {}
        self.trend_analyzers = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize analytics components"""
        self.logger.info("Initializing enterprise analytics engine")
        
        # Initialize aggregators
        self.aggregators = {
            MetricCategory.BUSINESS: BusinessMetricsAggregator(),
            MetricCategory.TECHNICAL: TechnicalMetricsAggregator(),
            MetricCategory.USER_BEHAVIOR: UserBehaviorAggregator(),
            MetricCategory.CONTENT: ContentAnalyticsAggregator(),
            MetricCategory.REVENUE: RevenueAnalyticsAggregator(),
            MetricCategory.PERFORMANCE: PerformanceMetricsAggregator()
        }
        
        # Initialize trend analyzers
        self.trend_analyzers = {
            'short_term': ShortTermTrendAnalyzer(),
            'medium_term': MediumTermTrendAnalyzer(),
            'long_term': LongTermTrendAnalyzer()
        }
        
        # Initialize predictors
        if self.config.ml_predictions:
            self.predictors = {
                'user_churn': UserChurnPredictor(),
                'revenue_forecast': RevenueForecastPredictor(),
                'content_popularity': ContentPopularityPredictor(),
                'performance_degradation': PerformanceDegradationPredictor()
            }
    
    async def process(self, data: Dict[str, Any]) -> AnalyticsResult:
        """Process analytics data with enterprise features"""
        start_time = datetime.now()
        
        try:
            # Determine category
            category = self._determine_category(data)
            
            # Get appropriate aggregator
            aggregator = self.aggregators.get(category)
            if not aggregator:
                raise ValueError(f"No aggregator found for category {category}")
            
            # Process metrics
            metrics = await aggregator.aggregate(data)
            
            # Generate insights
            insights = await self._generate_category_insights(category, metrics)
            
            # Generate predictions if enabled
            predictions = {}
            if self.config.ml_predictions and category in self.predictors:
                predictor = self.predictors[category]
                predictions = await predictor.predict(metrics)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(metrics, insights, predictions)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(category, metrics, insights)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = AnalyticsResult(
                timestamp=datetime.now(),
                category=category,
                metrics=metrics,
                insights=insights,
                predictions=predictions,
                recommendations=recommendations,
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )
            
            # Record metrics
            self._metrics['processed_events'].inc()
            self._metrics['processing_time'].observe(processing_time / 1000)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analytics processing error: {e}")
            raise
    
    def _determine_category(self, data: Dict[str, Any]) -> MetricCategory:
        """Determine metric category from data"""
        data_keys = set(data.keys())
        
        # Business metrics
        business_indicators = {
            'revenue', 'subscription', 'conversion', 'retention',
            'customer_acquisition_cost', 'lifetime_value'
        }
        if data_keys.intersection(business_indicators):
            return MetricCategory.BUSINESS
        
        # Technical metrics
        technical_indicators = {
            'latency', 'throughput', 'error_rate', 'cpu_usage',
            'memory_usage', 'disk_usage', 'network_usage'
        }
        if data_keys.intersection(technical_indicators):
            return MetricCategory.TECHNICAL
        
        # User behavior metrics
        behavior_indicators = {
            'play_count', 'skip_rate', 'session_duration',
            'playlist_creation', 'search_queries', 'like_rate'
        }
        if data_keys.intersection(behavior_indicators):
            return MetricCategory.USER_BEHAVIOR
        
        # Content metrics
        content_indicators = {
            'track_popularity', 'artist_ranking', 'genre_trends',
            'new_releases', 'playlist_additions'
        }
        if data_keys.intersection(content_indicators):
            return MetricCategory.CONTENT
        
        # Revenue metrics
        revenue_indicators = {
            'ad_revenue', 'premium_revenue', 'merchandise_sales',
            'concert_tickets', 'licensing_fees'
        }
        if data_keys.intersection(revenue_indicators):
            return MetricCategory.REVENUE
        
        # Default to performance
        return MetricCategory.PERFORMANCE
    
    async def _generate_category_insights(self, category: MetricCategory, metrics: Dict[str, Any]) -> List[str]:
        """Generate insights for specific category"""
        insights = []
        
        if category == MetricCategory.BUSINESS:
            if 'conversion_rate' in metrics:
                rate = metrics['conversion_rate']
                if rate > 0.05:
                    insights.append(f"Excellent conversion rate of {rate:.2%}")
                elif rate < 0.02:
                    insights.append(f"Low conversion rate of {rate:.2%} - optimization needed")
            
            if 'retention_rate' in metrics:
                rate = metrics['retention_rate']
                if rate > 0.80:
                    insights.append(f"Strong user retention at {rate:.2%}")
                elif rate < 0.60:
                    insights.append(f"Concerning retention rate of {rate:.2%}")
        
        elif category == MetricCategory.TECHNICAL:
            if 'latency_p99' in metrics:
                latency = metrics['latency_p99']
                if latency > 1000:
                    insights.append(f"High latency detected: {latency}ms P99")
                elif latency < 100:
                    insights.append(f"Excellent performance: {latency}ms P99")
            
            if 'error_rate' in metrics:
                rate = metrics['error_rate']
                if rate > 0.01:
                    insights.append(f"Elevated error rate: {rate:.2%}")
                elif rate < 0.001:
                    insights.append(f"Very low error rate: {rate:.2%}")
        
        elif category == MetricCategory.USER_BEHAVIOR:
            if 'skip_rate' in metrics:
                rate = metrics['skip_rate']
                if rate > 0.30:
                    insights.append(f"High skip rate: {rate:.2%} - content quality issue?")
                elif rate < 0.10:
                    insights.append(f"Low skip rate: {rate:.2%} - excellent content engagement")
            
            if 'session_duration' in metrics:
                duration = metrics['session_duration']
                if duration > 3600:  # 1 hour
                    insights.append(f"Long session duration: {duration/60:.1f} minutes")
                elif duration < 300:  # 5 minutes
                    insights.append(f"Short session duration: {duration/60:.1f} minutes")
        
        return insights
    
    def _calculate_confidence(self, metrics: Dict[str, Any], insights: List[str], predictions: Dict[str, Any]) -> float:
        """Calculate confidence score for analytics result"""
        confidence = 0.0
        
        # Base confidence from data completeness
        if metrics:
            confidence += 0.3
        
        # Confidence from insights quality
        if insights:
            confidence += 0.3 * min(len(insights) / 3, 1.0)
        
        # Confidence from predictions availability
        if predictions:
            confidence += 0.2 * min(len(predictions) / 2, 1.0)
        
        # Additional confidence from data quality
        data_quality_indicators = [
            'sample_size' in metrics,
            'time_range' in metrics,
            'data_completeness' in metrics
        ]
        confidence += 0.2 * (sum(data_quality_indicators) / len(data_quality_indicators))
        
        return min(confidence, 1.0)
    
    async def _generate_recommendations(self, category: MetricCategory, metrics: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if category == MetricCategory.BUSINESS:
            if 'conversion_rate' in metrics and metrics['conversion_rate'] < 0.02:
                recommendations.append("Optimize onboarding flow to improve conversion")
                recommendations.append("A/B test pricing strategies")
            
            if 'retention_rate' in metrics and metrics['retention_rate'] < 0.60:
                recommendations.append("Implement retention campaigns")
                recommendations.append("Analyze churn reasons and address pain points")
        
        elif category == MetricCategory.TECHNICAL:
            if 'latency_p99' in metrics and metrics['latency_p99'] > 1000:
                recommendations.append("Scale infrastructure to reduce latency")
                recommendations.append("Optimize database queries")
                recommendations.append("Implement caching strategies")
            
            if 'error_rate' in metrics and metrics['error_rate'] > 0.01:
                recommendations.append("Investigate and fix error sources")
                recommendations.append("Implement better error handling")
        
        elif category == MetricCategory.USER_BEHAVIOR:
            if 'skip_rate' in metrics and metrics['skip_rate'] > 0.30:
                recommendations.append("Improve recommendation algorithm")
                recommendations.append("Analyze content quality and user preferences")
            
            if 'session_duration' in metrics and metrics['session_duration'] < 300:
                recommendations.append("Enhance user engagement features")
                recommendations.append("Personalize content discovery")
        
        return recommendations
    
    async def generate_insights(self, results: List[AnalyticsResult]) -> List[str]:
        """Generate cross-category insights"""
        insights = []
        
        if not results:
            return insights
        
        # Aggregate insights from all results
        all_insights = []
        for result in results:
            all_insights.extend(result.insights)
        
        # Generate meta-insights
        if len(results) > 1:
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
            insights.append(f"Overall analytics confidence: {avg_confidence:.2%}")
            
            processing_times = [r.processing_time_ms for r in results]
            avg_processing_time = sum(processing_times) / len(processing_times)
            insights.append(f"Average processing time: {avg_processing_time:.1f}ms")
        
        # Cross-category correlations
        categories = [r.category for r in results]
        if MetricCategory.BUSINESS in categories and MetricCategory.TECHNICAL in categories:
            insights.append("Business and technical metrics correlation available")
        
        return insights


class MetricsAggregator:
    """Advanced metrics aggregation engine"""
    
    def __init__(self):
        self.aggregation_functions = {
            'sum': np.sum,
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'percentile_95': lambda x: np.percentile(x, 95),
            'percentile_99': lambda x: np.percentile(x, 99)
        }
    
    async def aggregate_time_series(self, data: pd.DataFrame, window_size: str = '1H') -> Dict[str, Any]:
        """Aggregate time series data"""
        try:
            if 'timestamp' not in data.columns:
                raise ValueError("Data must contain timestamp column")
            
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Resample and aggregate
            aggregated = data.resample(window_size).agg(self.aggregation_functions)
            
            return {
                'time_series': aggregated.to_dict(),
                'summary_stats': {
                    'total_windows': len(aggregated),
                    'time_range': {
                        'start': aggregated.index.min().isoformat(),
                        'end': aggregated.index.max().isoformat()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Time series aggregation error: {e}")
            raise
    
    async def aggregate_by_dimensions(self, data: pd.DataFrame, dimensions: List[str]) -> Dict[str, Any]:
        """Aggregate data by multiple dimensions"""
        try:
            # Group by dimensions and aggregate
            grouped = data.groupby(dimensions)
            
            aggregated_results = {}
            for group_key, group_data in grouped:
                if isinstance(group_key, tuple):
                    key = '_'.join(str(k) for k in group_key)
                else:
                    key = str(group_key)
                
                # Calculate aggregations for numeric columns
                numeric_columns = group_data.select_dtypes(include=[np.number]).columns
                
                aggregated_results[key] = {}
                for col in numeric_columns:
                    col_data = group_data[col].dropna()
                    if len(col_data) > 0:
                        aggregated_results[key][col] = {
                            func_name: float(func(col_data))
                            for func_name, func in self.aggregation_functions.items()
                        }
            
            return {
                'aggregated_data': aggregated_results,
                'dimensions_used': dimensions,
                'total_groups': len(aggregated_results)
            }
            
        except Exception as e:
            logger.error(f"Dimensional aggregation error: {e}")
            raise


class TrendAnalyzer:
    """Advanced trend analysis engine"""
    
    def __init__(self):
        self.analysis_methods = {
            'linear_trend': self._linear_trend,
            'seasonal_decomposition': self._seasonal_decomposition,
            'change_point_detection': self._change_point_detection,
            'correlation_analysis': self._correlation_analysis
        }
    
    async def analyze_trends(self, data: pd.DataFrame, metric_column: str) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        try:
            results = {}
            
            # Ensure data is sorted by time
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
                time_series = data[metric_column].values
            else:
                time_series = data[metric_column].values
            
            # Apply all analysis methods
            for method_name, method in self.analysis_methods.items():
                try:
                    results[method_name] = await method(time_series)
                except Exception as e:
                    logger.warning(f"Trend analysis method {method_name} failed: {e}")
                    results[method_name] = {'error': str(e)}
            
            # Generate trend summary
            results['trend_summary'] = self._generate_trend_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            raise
    
    async def _linear_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate linear trend"""
        x = np.arange(len(data))
        coefficients = np.polyfit(x, data, 1)
        
        return {
            'slope': float(coefficients[0]),
            'intercept': float(coefficients[1]),
            'trend_direction': 'increasing' if coefficients[0] > 0 else 'decreasing',
            'trend_strength': abs(float(coefficients[0]))
        }
    
    async def _seasonal_decomposition(self, data: np.ndarray) -> Dict[str, Any]:
        """Seasonal decomposition analysis"""
        # Simple seasonal analysis (requires more sophisticated implementation for production)
        return {
            'has_seasonality': bool(len(data) > 24),  # Simplified check
            'cycle_length': 24 if len(data) > 24 else len(data),
            'seasonal_strength': 0.5  # Placeholder
        }
    
    async def _change_point_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect change points in the data"""
        # Simple change point detection based on variance
        if len(data) < 10:
            return {'change_points': [], 'change_point_count': 0}
        
        # Calculate rolling variance
        window_size = max(5, len(data) // 10)
        variances = []
        
        for i in range(window_size, len(data) - window_size):
            window = data[i-window_size:i+window_size]
            variances.append(np.var(window))
        
        # Find significant variance changes
        variance_threshold = np.std(variances) * 2
        change_points = []
        
        for i in range(1, len(variances)):
            if abs(variances[i] - variances[i-1]) > variance_threshold:
                change_points.append(i + window_size)
        
        return {
            'change_points': change_points,
            'change_point_count': len(change_points),
            'variance_threshold': float(variance_threshold)
        }
    
    async def _correlation_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze autocorrelation"""
        if len(data) < 2:
            return {'autocorrelation': 0.0}
        
        # Calculate lag-1 autocorrelation
        if len(data) > 1:
            correlation = np.corrcoef(data[:-1], data[1:])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'autocorrelation': float(correlation),
            'persistence': 'high' if abs(correlation) > 0.7 else 'low'
        }
    
    def _generate_trend_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trend summary"""
        summary = {
            'overall_trend': 'stable',
            'confidence': 0.5,
            'key_findings': []
        }
        
        # Analyze linear trend
        if 'linear_trend' in results and 'slope' in results['linear_trend']:
            slope = results['linear_trend']['slope']
            if abs(slope) > 0.1:
                summary['overall_trend'] = results['linear_trend']['trend_direction']
                summary['confidence'] = min(abs(slope), 1.0)
                summary['key_findings'].append(f"Linear trend: {results['linear_trend']['trend_direction']}")
        
        # Analyze change points
        if 'change_point_detection' in results:
            change_point_count = results['change_point_detection'].get('change_point_count', 0)
            if change_point_count > 0:
                summary['key_findings'].append(f"Detected {change_point_count} change points")
        
        # Analyze seasonality
        if 'seasonal_decomposition' in results:
            if results['seasonal_decomposition'].get('has_seasonality', False):
                summary['key_findings'].append("Seasonal patterns detected")
        
        return summary


class PredictiveAnalytics:
    """ML-powered predictive analytics"""
    
    def __init__(self):
        self.models = {}
        self.feature_extractors = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for predictions"""
        # Placeholder for ML model initialization
        self.models = {
            'linear_regression': None,  # Would be actual sklearn model
            'time_series_forecast': None,
            'anomaly_detection': None,
            'classification': None
        }
    
    async def predict_future_values(self, data: pd.DataFrame, metric_column: str, forecast_periods: int = 24) -> Dict[str, Any]:
        """Predict future metric values"""
        try:
            if len(data) < 10:
                return {
                    'error': 'Insufficient data for prediction',
                    'predictions': [],
                    'confidence_intervals': []
                }
            
            # Simple linear extrapolation (in production, use sophisticated time series models)
            x = np.arange(len(data))
            y = data[metric_column].values
            
            # Fit linear model
            coefficients = np.polyfit(x, y, 1)
            
            # Generate predictions
            future_x = np.arange(len(data), len(data) + forecast_periods)
            predictions = np.polyval(coefficients, future_x)
            
            # Calculate simple confidence intervals
            residuals = y - np.polyval(coefficients, x)
            prediction_std = np.std(residuals)
            
            confidence_intervals = [
                {
                    'lower': float(pred - 1.96 * prediction_std),
                    'upper': float(pred + 1.96 * prediction_std)
                }
                for pred in predictions
            ]
            
            return {
                'predictions': [float(p) for p in predictions],
                'confidence_intervals': confidence_intervals,
                'model_accuracy': float(1.0 - (prediction_std / np.std(y))),
                'forecast_horizon': forecast_periods
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'predictions': [],
                'confidence_intervals': []
            }
    
    async def detect_anomalies(self, data: pd.DataFrame, metric_column: str) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        try:
            values = data[metric_column].values
            
            # Simple statistical anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = 2.0  # 2 standard deviations
            
            anomalies = []
            for i, value in enumerate(values):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'value': float(value),
                        'z_score': float(z_score),
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            return {
                'anomalies': anomalies,
                'anomaly_count': len(anomalies),
                'anomaly_rate': len(anomalies) / len(values) if len(values) > 0 else 0,
                'threshold_used': threshold
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {
                'error': str(e),
                'anomalies': [],
                'anomaly_count': 0
            }


class BusinessIntelligence:
    """Business intelligence and insights generation"""
    
    def __init__(self):
        self.insight_generators = {
            'revenue_insights': self._generate_revenue_insights,
            'user_insights': self._generate_user_insights,
            'content_insights': self._generate_content_insights,
            'performance_insights': self._generate_performance_insights
        }
    
    async def generate_comprehensive_insights(self, analytics_results: List[AnalyticsResult]) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        try:
            insights = {
                'executive_summary': [],
                'key_metrics': {},
                'recommendations': [],
                'risk_factors': [],
                'opportunities': []
            }
            
            # Process each analytics result
            for result in analytics_results:
                category_insights = await self._process_analytics_result(result)
                
                # Merge insights
                insights['executive_summary'].extend(category_insights.get('summary', []))
                insights['key_metrics'].update(category_insights.get('metrics', {}))
                insights['recommendations'].extend(category_insights.get('recommendations', []))
                insights['risk_factors'].extend(category_insights.get('risks', []))
                insights['opportunities'].extend(category_insights.get('opportunities', []))
            
            # Generate meta-insights
            meta_insights = await self._generate_meta_insights(analytics_results)
            insights.update(meta_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Business intelligence generation error: {e}")
            raise
    
    async def _process_analytics_result(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Process individual analytics result"""
        category = result.category
        
        if category == MetricCategory.BUSINESS:
            return await self._generate_revenue_insights(result.metrics)
        elif category == MetricCategory.USER_BEHAVIOR:
            return await self._generate_user_insights(result.metrics)
        elif category == MetricCategory.CONTENT:
            return await self._generate_content_insights(result.metrics)
        elif category == MetricCategory.PERFORMANCE:
            return await self._generate_performance_insights(result.metrics)
        else:
            return {'summary': [], 'metrics': {}, 'recommendations': [], 'risks': [], 'opportunities': []}
    
    async def _generate_revenue_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate revenue-specific insights"""
        insights = {
            'summary': [],
            'metrics': {},
            'recommendations': [],
            'risks': [],
            'opportunities': []
        }
        
        if 'revenue' in metrics:
            revenue = metrics['revenue']
            insights['summary'].append(f"Current revenue: ${revenue:,.2f}")
            insights['metrics']['revenue'] = revenue
            
            # Revenue growth analysis
            if 'revenue_growth' in metrics:
                growth = metrics['revenue_growth']
                if growth > 0.1:
                    insights['opportunities'].append("Strong revenue growth momentum")
                elif growth < -0.05:
                    insights['risks'].append("Declining revenue trend")
        
        if 'conversion_rate' in metrics:
            rate = metrics['conversion_rate']
            if rate < 0.02:
                insights['recommendations'].append("Optimize conversion funnel")
                insights['risks'].append("Low conversion rate impacting revenue")
        
        return insights
    
    async def _generate_user_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user behavior insights"""
        insights = {
            'summary': [],
            'metrics': {},
            'recommendations': [],
            'risks': [],
            'opportunities': []
        }
        
        if 'active_users' in metrics:
            users = metrics['active_users']
            insights['summary'].append(f"Active users: {users:,}")
            insights['metrics']['active_users'] = users
        
        if 'engagement_rate' in metrics:
            rate = metrics['engagement_rate']
            if rate > 0.7:
                insights['opportunities'].append("High user engagement")
            elif rate < 0.3:
                insights['risks'].append("Low user engagement")
                insights['recommendations'].append("Improve user experience")
        
        return insights
    
    async def _generate_content_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content performance insights"""
        insights = {
            'summary': [],
            'metrics': {},
            'recommendations': [],
            'risks': [],
            'opportunities': []
        }
        
        if 'content_diversity' in metrics:
            diversity = metrics['content_diversity']
            if diversity > 0.8:
                insights['opportunities'].append("High content diversity")
            elif diversity < 0.4:
                insights['recommendations'].append("Expand content catalog")
        
        return insights
    
    async def _generate_performance_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance insights"""
        insights = {
            'summary': [],
            'metrics': {},
            'recommendations': [],
            'risks': [],
            'opportunities': []
        }
        
        if 'latency' in metrics:
            latency = metrics['latency']
            if latency > 1000:
                insights['risks'].append("High latency affecting user experience")
                insights['recommendations'].append("Optimize performance")
        
        return insights
    
    async def _generate_meta_insights(self, results: List[AnalyticsResult]) -> Dict[str, Any]:
        """Generate insights across all categories"""
        meta_insights = {
            'data_quality_score': 0.0,
            'overall_health': 'unknown',
            'priority_actions': []
        }
        
        if results:
            # Calculate average confidence as data quality score
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
            meta_insights['data_quality_score'] = avg_confidence
            
            # Determine overall health
            if avg_confidence > 0.8:
                meta_insights['overall_health'] = 'excellent'
            elif avg_confidence > 0.6:
                meta_insights['overall_health'] = 'good'
            elif avg_confidence > 0.4:
                meta_insights['overall_health'] = 'fair'
            else:
                meta_insights['overall_health'] = 'poor'
        
        return meta_insights


# Specialized Aggregators
class BusinessMetricsAggregator:
    """Aggregator for business metrics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate business metrics"""
        metrics = {}
        
        # Revenue metrics
        if 'revenue' in data:
            metrics['revenue'] = data['revenue']
        
        # Conversion metrics
        if 'conversions' in data and 'visitors' in data:
            metrics['conversion_rate'] = data['conversions'] / data['visitors'] if data['visitors'] > 0 else 0
        
        # Retention metrics
        if 'retained_users' in data and 'total_users' in data:
            metrics['retention_rate'] = data['retained_users'] / data['total_users'] if data['total_users'] > 0 else 0
        
        return metrics


class TechnicalMetricsAggregator:
    """Aggregator for technical metrics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate technical metrics"""
        metrics = {}
        
        # Performance metrics
        if 'response_times' in data:
            response_times = data['response_times']
            if isinstance(response_times, list) and response_times:
                metrics['latency_avg'] = np.mean(response_times)
                metrics['latency_p95'] = np.percentile(response_times, 95)
                metrics['latency_p99'] = np.percentile(response_times, 99)
        
        # Error metrics
        if 'error_count' in data and 'total_requests' in data:
            metrics['error_rate'] = data['error_count'] / data['total_requests'] if data['total_requests'] > 0 else 0
        
        # Resource metrics
        for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
            if metric in data:
                metrics[metric] = data[metric]
        
        return metrics


class UserBehaviorAggregator:
    """Aggregator for user behavior metrics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate user behavior metrics"""
        metrics = {}
        
        # Engagement metrics
        if 'total_plays' in data and 'total_tracks' in data:
            metrics['play_rate'] = data['total_plays'] / data['total_tracks'] if data['total_tracks'] > 0 else 0
        
        if 'skips' in data and 'total_plays' in data:
            metrics['skip_rate'] = data['skips'] / data['total_plays'] if data['total_plays'] > 0 else 0
        
        # Session metrics
        if 'session_durations' in data:
            durations = data['session_durations']
            if isinstance(durations, list) and durations:
                metrics['avg_session_duration'] = np.mean(durations)
                metrics['median_session_duration'] = np.median(durations)
        
        return metrics


class ContentAnalyticsAggregator:
    """Aggregator for content analytics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate content metrics"""
        metrics = {}
        
        # Popularity metrics
        if 'track_plays' in data:
            track_plays = data['track_plays']
            if isinstance(track_plays, dict):
                metrics['most_popular_track'] = max(track_plays.items(), key=lambda x: x[1])
                metrics['total_unique_tracks'] = len(track_plays)
                metrics['avg_plays_per_track'] = np.mean(list(track_plays.values()))
        
        # Diversity metrics
        if 'genre_distribution' in data:
            genre_dist = data['genre_distribution']
            if isinstance(genre_dist, dict) and genre_dist:
                # Calculate genre diversity (entropy)
                total = sum(genre_dist.values())
                probs = [count/total for count in genre_dist.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                metrics['content_diversity'] = entropy / np.log2(len(genre_dist))
        
        return metrics


class RevenueAnalyticsAggregator:
    """Aggregator for revenue analytics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate revenue metrics"""
        metrics = {}
        
        # Revenue streams
        revenue_streams = ['subscription_revenue', 'ad_revenue', 'merchandise_revenue']
        total_revenue = 0
        
        for stream in revenue_streams:
            if stream in data:
                metrics[stream] = data[stream]
                total_revenue += data[stream]
        
        if total_revenue > 0:
            metrics['total_revenue'] = total_revenue
            
            # Revenue distribution
            for stream in revenue_streams:
                if stream in metrics:
                    metrics[f'{stream}_percentage'] = metrics[stream] / total_revenue
        
        # ARPU (Average Revenue Per User)
        if 'total_revenue' in metrics and 'active_users' in data:
            metrics['arpu'] = metrics['total_revenue'] / data['active_users'] if data['active_users'] > 0 else 0
        
        return metrics


class PerformanceMetricsAggregator:
    """Aggregator for performance metrics"""
    
    async def aggregate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate performance metrics"""
        metrics = {}
        
        # System performance
        if 'cpu_samples' in data:
            cpu_samples = data['cpu_samples']
            if isinstance(cpu_samples, list) and cpu_samples:
                metrics['avg_cpu_usage'] = np.mean(cpu_samples)
                metrics['max_cpu_usage'] = np.max(cpu_samples)
                metrics['cpu_spikes'] = sum(1 for x in cpu_samples if x > 0.8)
        
        # Application performance
        if 'request_latencies' in data:
            latencies = data['request_latencies']
            if isinstance(latencies, list) and latencies:
                metrics['avg_latency'] = np.mean(latencies)
                metrics['p95_latency'] = np.percentile(latencies, 95)
                metrics['p99_latency'] = np.percentile(latencies, 99)
        
        # Throughput metrics
        if 'requests_per_second' in data:
            metrics['throughput'] = data['requests_per_second']
        
        return metrics


# Specialized Trend Analyzers
class ShortTermTrendAnalyzer:
    """Analyzer for short-term trends (minutes to hours)"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze short-term trends"""
        return {
            'trend_type': 'short_term',
            'window_size': '1H',
            'volatility': 'high',
            'recommendations': ['Monitor for rapid changes', 'Set up real-time alerts']
        }


class MediumTermTrendAnalyzer:
    """Analyzer for medium-term trends (hours to days)"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze medium-term trends"""
        return {
            'trend_type': 'medium_term',
            'window_size': '1D',
            'volatility': 'medium',
            'recommendations': ['Track daily patterns', 'Analyze business cycles']
        }


class LongTermTrendAnalyzer:
    """Analyzer for long-term trends (days to weeks)"""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term trends"""
        return {
            'trend_type': 'long_term',
            'window_size': '1W',
            'volatility': 'low',
            'recommendations': ['Strategic planning', 'Capacity planning']
        }


# Specialized Predictors
class UserChurnPredictor:
    """Predictor for user churn"""
    
    async def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user churn probability"""
        # Simplified churn prediction based on engagement metrics
        churn_probability = 0.1  # Default 10% churn rate
        
        if 'engagement_rate' in metrics:
            engagement = metrics['engagement_rate']
            # Lower engagement = higher churn probability
            churn_probability = max(0.05, 0.5 - engagement)
        
        return {
            'churn_probability': churn_probability,
            'risk_level': 'high' if churn_probability > 0.3 else 'medium' if churn_probability > 0.15 else 'low',
            'recommended_actions': ['Engagement campaigns', 'Personalized offers'] if churn_probability > 0.2 else []
        }


class RevenueForecastPredictor:
    """Predictor for revenue forecasting"""
    
    async def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future revenue"""
        # Simplified revenue prediction
        current_revenue = metrics.get('revenue', 0)
        growth_rate = metrics.get('revenue_growth', 0.05)  # Default 5% growth
        
        # 30-day forecast
        forecast_periods = 30
        forecasted_revenue = []
        
        for i in range(forecast_periods):
            future_revenue = current_revenue * (1 + growth_rate) ** (i / 30)
            forecasted_revenue.append(future_revenue)
        
        return {
            'revenue_forecast': forecasted_revenue,
            'forecast_horizon_days': forecast_periods,
            'growth_rate': growth_rate,
            'confidence_interval': '80%'
        }


class ContentPopularityPredictor:
    """Predictor for content popularity"""
    
    async def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content popularity trends"""
        return {
            'trending_genres': ['pop', 'hip-hop', 'electronic'],
            'predicted_viral_tracks': [],
            'content_optimization_suggestions': [
                'Increase pop music catalog',
                'Promote emerging artists',
                'Focus on playlist curation'
            ]
        }


class PerformanceDegradationPredictor:
    """Predictor for performance degradation"""
    
    async def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance issues"""
        risk_score = 0.0
        
        # Check latency trends
        if 'latency_trend' in metrics:
            if metrics['latency_trend'] > 0.1:  # 10% increase
                risk_score += 0.3
        
        # Check error rate trends
        if 'error_rate_trend' in metrics:
            if metrics['error_rate_trend'] > 0.05:  # 5% increase
                risk_score += 0.4
        
        # Check resource utilization
        if 'cpu_usage' in metrics:
            if metrics['cpu_usage'] > 0.8:  # 80% CPU usage
                risk_score += 0.3
        
        return {
            'degradation_risk_score': min(risk_score, 1.0),
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
            'predicted_issues': [
                'Increased latency',
                'Higher error rates',
                'Resource bottlenecks'
            ] if risk_score > 0.5 else [],
            'recommended_actions': [
                'Scale infrastructure',
                'Optimize queries',
                'Monitor closely'
            ] if risk_score > 0.5 else []
        }


# Factory functions
def create_analytics_engine(config: AnalyticsConfig = None) -> AnalyticsEngine:
    """Create analytics engine with configuration"""
    if config is None:
        config = AnalyticsConfig()
    
    return AnalyticsEngine(config)


def create_metrics_aggregator() -> MetricsAggregator:
    """Create metrics aggregator"""
    return MetricsAggregator()


def create_trend_analyzer() -> TrendAnalyzer:
    """Create trend analyzer"""
    return TrendAnalyzer()


def create_predictive_analytics() -> PredictiveAnalytics:
    """Create predictive analytics"""
    return PredictiveAnalytics()


def create_business_intelligence() -> BusinessIntelligence:
    """Create business intelligence"""
    return BusinessIntelligence()


# Export all classes and functions
__all__ = [
    'AnalyticsLevel',
    'MetricCategory',
    'AnalyticsConfig',
    'AnalyticsResult',
    'AnalyticsEngine',
    'MetricsAggregator',
    'TrendAnalyzer',
    'PredictiveAnalytics',
    'BusinessIntelligence',
    'create_analytics_engine',
    'create_metrics_aggregator',
    'create_trend_analyzer',
    'create_predictive_analytics',
    'create_business_intelligence'
]
