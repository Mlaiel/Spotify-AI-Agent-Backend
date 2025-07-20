"""
Spotify AI Agent - Streaming Data Formatters
===========================================

Ultra-advanced real-time streaming data formatting system for live analytics,
streaming metrics, and real-time business intelligence dashboards.

This module handles real-time formatting for:
- Live streaming analytics and metrics
- Real-time user behavior tracking
- Live recommendation performance
- Real-time revenue and engagement tracking
- Streaming ML model performance monitoring
- Live content performance analytics
- Real-time geographic and demographic insights

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import deque, defaultdict
import statistics

logger = structlog.get_logger(__name__)


class StreamingMetricType(Enum):
    """Types of streaming metrics."""
    REAL_TIME = "real_time"
    WINDOWED = "windowed"
    CUMULATIVE = "cumulative"
    DERIVED = "derived"


class WindowType(Enum):
    """Time window types for aggregations."""
    TUMBLING = "tumbling"      # Non-overlapping fixed windows
    SLIDING = "sliding"        # Overlapping windows
    SESSION = "session"        # Event-driven windows
    HOPPING = "hopping"        # Fixed interval overlapping windows


class AggregationType(Enum):
    """Aggregation types for streaming data."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"
    PERCENTILE = "percentile"
    RATE = "rate"
    DERIVATIVE = "derivative"


@dataclass
class StreamingDataPoint:
    """Individual streaming data point."""
    
    timestamp: datetime
    metric_name: str
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class StreamingWindow:
    """Streaming window configuration."""
    
    window_type: WindowType
    size_seconds: int
    slide_seconds: Optional[int] = None
    aggregation: AggregationType = AggregationType.SUM
    
    def __post_init__(self):
        if self.window_type == WindowType.SLIDING and self.slide_seconds is None:
            self.slide_seconds = self.size_seconds // 2


@dataclass
class FormattedStreamingData:
    """Container for formatted streaming data."""
    
    timestamp: datetime
    formatted_data: str
    raw_data: List[StreamingDataPoint]
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "formatted_data": self.formatted_data,
            "raw_data": [dp.to_dict() for dp in self.raw_data],
            "aggregated_metrics": self.aggregated_metrics,
            "metadata": self.metadata
        }


class BaseStreamingFormatter:
    """Base class for streaming data formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
        # Streaming configuration
        self.buffer_size = config.get('buffer_size', 1000)
        self.flush_interval = config.get('flush_interval', 5.0)  # seconds
        self.window_configs = config.get('windows', [])
        
        # Data buffers
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.windows = {}
        self.last_flush = time.time()
        
        # Initialize windows
        for window_config in self.window_configs:
            window = StreamingWindow(**window_config)
            self.windows[f"{window.window_type.value}_{window.size_seconds}"] = {
                'config': window,
                'data': deque(),
                'last_slide': time.time()
            }
    
    async def add_data_point(self, data_point: StreamingDataPoint) -> None:
        """Add a data point to the streaming buffer."""
        self.data_buffer.append(data_point)
        
        # Add to windows
        for window_id, window_data in self.windows.items():
            window_data['data'].append(data_point)
            await self._manage_window(window_id, window_data)
        
        # Check if we need to flush
        if self._should_flush():
            await self.flush()
    
    async def _manage_window(self, window_id: str, window_data: Dict[str, Any]) -> None:
        """Manage window data based on window type."""
        window_config = window_data['config']
        current_time = time.time()
        
        if window_config.window_type == WindowType.TUMBLING:
            # Remove data older than window size
            cutoff_time = current_time - window_config.size_seconds
            while (window_data['data'] and 
                   window_data['data'][0].timestamp.timestamp() < cutoff_time):
                window_data['data'].popleft()
        
        elif window_config.window_type == WindowType.SLIDING:
            # Slide window based on slide interval
            if current_time - window_data['last_slide'] >= window_config.slide_seconds:
                cutoff_time = current_time - window_config.size_seconds
                while (window_data['data'] and 
                       window_data['data'][0].timestamp.timestamp() < cutoff_time):
                    window_data['data'].popleft()
                window_data['last_slide'] = current_time
    
    def _should_flush(self) -> bool:
        """Check if we should flush the buffer."""
        return (time.time() - self.last_flush) >= self.flush_interval
    
    async def flush(self) -> List[FormattedStreamingData]:
        """Flush and format current buffer data."""
        if not self.data_buffer:
            return []
        
        current_data = list(self.data_buffer)
        self.data_buffer.clear()
        self.last_flush = time.time()
        
        # Format the data
        formatted_results = await self._format_streaming_data(current_data)
        
        return formatted_results
    
    async def _format_streaming_data(self, data_points: List[StreamingDataPoint]) -> List[FormattedStreamingData]:
        """Format streaming data points - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _format_streaming_data")
    
    def calculate_aggregation(self, data_points: List[StreamingDataPoint], 
                            aggregation: AggregationType) -> float:
        """Calculate aggregation for data points."""
        if not data_points:
            return 0.0
        
        values = [dp.value for dp in data_points]
        
        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return len(values)
        elif aggregation == AggregationType.DISTINCT_COUNT:
            return len(set(values))
        elif aggregation == AggregationType.PERCENTILE:
            # Default to 95th percentile
            return statistics.quantiles(values, n=20)[18]  # 95th percentile
        elif aggregation == AggregationType.RATE:
            if len(data_points) < 2:
                return 0.0
            time_diff = (data_points[-1].timestamp - data_points[0].timestamp).total_seconds()
            return sum(values) / max(time_diff, 1.0)
        
        return 0.0


class RealTimeAnalyticsFormatter(BaseStreamingFormatter):
    """Real-time analytics formatter for live dashboard updates."""
    
    async def _format_streaming_data(self, data_points: List[StreamingDataPoint]) -> List[FormattedStreamingData]:
        """Format real-time analytics data."""
        
        if not data_points:
            return []
        
        # Group data points by metric type
        metrics_by_type = defaultdict(list)
        for dp in data_points:
            metrics_by_type[dp.metric_name].append(dp)
        
        formatted_results = []
        timestamp = datetime.now(timezone.utc)
        
        # Format each metric type
        for metric_name, metric_data in metrics_by_type.items():
            formatted_data = await self._format_metric_group(metric_name, metric_data)
            
            # Calculate aggregated metrics
            aggregated_metrics = await self._calculate_aggregated_metrics(metric_data)
            
            # Create metadata
            metadata = {
                "metric_name": metric_name,
                "data_points_count": len(metric_data),
                "time_range": {
                    "start": min(dp.timestamp for dp in metric_data).isoformat(),
                    "end": max(dp.timestamp for dp in metric_data).isoformat()
                },
                "tenant_id": self.tenant_id,
                "formatter_type": "real_time_analytics"
            }
            
            result = FormattedStreamingData(
                timestamp=timestamp,
                formatted_data=formatted_data,
                raw_data=metric_data,
                aggregated_metrics=aggregated_metrics,
                metadata=metadata
            )
            
            formatted_results.append(result)
        
        return formatted_results
    
    async def _format_metric_group(self, metric_name: str, data_points: List[StreamingDataPoint]) -> str:
        """Format a group of metrics into dashboard format."""
        
        latest_value = data_points[-1].value if data_points else 0
        timestamp = data_points[-1].timestamp if data_points else datetime.now(timezone.utc)
        
        # Calculate trend
        trend = await self._calculate_trend(data_points)
        trend_indicator = "📈" if trend > 0 else "📉" if trend < 0 else "➖"
        
        # Format based on metric type
        if "revenue" in metric_name.lower():
            formatted_value = f"${latest_value:,.2f}"
        elif "rate" in metric_name.lower() or "percentage" in metric_name.lower():
            formatted_value = f"{latest_value:.2f}%"
        elif "count" in metric_name.lower():
            formatted_value = f"{latest_value:,}"
        else:
            formatted_value = f"{latest_value:.2f}"
        
        # Create real-time dashboard format
        dashboard_format = f"""
🔴 **LIVE**: {metric_name.replace('_', ' ').title()}
**Current**: {formatted_value} {trend_indicator}
**Trend**: {trend:+.2f}%
**Updated**: {timestamp.strftime('%H:%M:%S UTC')}
**Status**: {'🟢 Normal' if abs(trend) < 10 else '🟡 Attention' if abs(trend) < 25 else '🔴 Alert'}
        """.strip()
        
        return dashboard_format
    
    async def _calculate_trend(self, data_points: List[StreamingDataPoint]) -> float:
        """Calculate trend percentage for the data points."""
        if len(data_points) < 2:
            return 0.0
        
        # Use first and last values for trend calculation
        first_value = data_points[0].value
        last_value = data_points[-1].value
        
        if first_value == 0:
            return 100.0 if last_value > 0 else 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    async def _calculate_aggregated_metrics(self, data_points: List[StreamingDataPoint]) -> Dict[str, Any]:
        """Calculate aggregated metrics for the data points."""
        
        if not data_points:
            return {}
        
        values = [dp.value for dp in data_points]
        
        aggregated = {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "trend": await self._calculate_trend(data_points)
        }
        
        # Add percentiles if we have enough data
        if len(values) >= 4:
            aggregated.update({
                "p50": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            })
        
        return aggregated


class LiveUserMetricsFormatter(BaseStreamingFormatter):
    """Formatter for live user behavior and engagement metrics."""
    
    async def _format_streaming_data(self, data_points: List[StreamingDataPoint]) -> List[FormattedStreamingData]:
        """Format live user metrics data."""
        
        if not data_points:
            return []
        
        # Categorize user metrics
        user_metrics = {
            'active_users': [],
            'engagement': [],
            'sessions': [],
            'content_interaction': []
        }
        
        for dp in data_points:
            if 'active_user' in dp.metric_name:
                user_metrics['active_users'].append(dp)
            elif 'engagement' in dp.metric_name or 'interaction' in dp.metric_name:
                user_metrics['engagement'].append(dp)
            elif 'session' in dp.metric_name:
                user_metrics['sessions'].append(dp)
            elif 'play' in dp.metric_name or 'skip' in dp.metric_name or 'save' in dp.metric_name:
                user_metrics['content_interaction'].append(dp)
        
        formatted_results = []
        timestamp = datetime.now(timezone.utc)
        
        # Format user engagement dashboard
        engagement_dashboard = await self._format_user_engagement_dashboard(user_metrics)
        
        # Calculate user behavior insights
        behavior_insights = await self._analyze_user_behavior(user_metrics)
        
        # Create comprehensive user metrics report
        formatted_data = f"""
🎵 **SPOTIFY AI - LIVE USER METRICS** 🎵

{engagement_dashboard}

## 🔍 Real-Time Insights
{chr(10).join(f"• {insight}" for insight in behavior_insights)}

## 📊 Live Statistics
**Total Active Sessions**: {len(user_metrics['sessions']):,}
**Avg Engagement Rate**: {self._calculate_avg_engagement(user_metrics['engagement']):.1f}%
**Content Interactions/Min**: {self._calculate_interaction_rate(user_metrics['content_interaction']):.0f}

*Updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*
        """.strip()
        
        # Aggregate all metrics
        all_aggregated = {}
        for category, category_data in user_metrics.items():
            if category_data:
                all_aggregated[category] = await self._calculate_aggregated_metrics(category_data)
        
        metadata = {
            "formatter_type": "live_user_metrics",
            "categories_count": len([k for k, v in user_metrics.items() if v]),
            "total_data_points": len(data_points),
            "tenant_id": self.tenant_id,
            "generated_at": timestamp.isoformat()
        }
        
        result = FormattedStreamingData(
            timestamp=timestamp,
            formatted_data=formatted_data,
            raw_data=data_points,
            aggregated_metrics=all_aggregated,
            metadata=metadata
        )
        
        formatted_results.append(result)
        return formatted_results
    
    async def _format_user_engagement_dashboard(self, user_metrics: Dict[str, List[StreamingDataPoint]]) -> str:
        """Format user engagement dashboard section."""
        
        dashboard_sections = []
        
        # Active Users Section
        if user_metrics['active_users']:
            active_users = user_metrics['active_users'][-1].value if user_metrics['active_users'] else 0
            trend = await self._calculate_trend(user_metrics['active_users'])
            trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➖"
            
            dashboard_sections.append(f"""
### 👥 Active Users
**Current**: {active_users:,.0f} users {trend_icon}
**Trend**: {trend:+.1f}%
            """.strip())
        
        # Engagement Metrics
        if user_metrics['engagement']:
            avg_engagement = self._calculate_avg_engagement(user_metrics['engagement'])
            engagement_trend = await self._calculate_trend(user_metrics['engagement'])
            engagement_icon = "🟢" if avg_engagement > 70 else "🟡" if avg_engagement > 50 else "🔴"
            
            dashboard_sections.append(f"""
### 💝 Engagement Metrics
**Avg Engagement**: {avg_engagement:.1f}% {engagement_icon}
**Trend**: {engagement_trend:+.1f}%
            """.strip())
        
        # Session Activity
        if user_metrics['sessions']:
            active_sessions = len(user_metrics['sessions'])
            avg_session_length = statistics.mean([dp.value for dp in user_metrics['sessions']]) if user_metrics['sessions'] else 0
            
            dashboard_sections.append(f"""
### 🎧 Session Activity
**Active Sessions**: {active_sessions:,}
**Avg Session Length**: {avg_session_length:.1f} minutes
            """.strip())
        
        # Content Interactions
        if user_metrics['content_interaction']:
            interaction_rate = self._calculate_interaction_rate(user_metrics['content_interaction'])
            
            dashboard_sections.append(f"""
### 🎵 Content Interactions
**Interactions/Minute**: {interaction_rate:.0f}
**Total Interactions**: {len(user_metrics['content_interaction']):,}
            """.strip())
        
        return "\n\n".join(dashboard_sections)
    
    def _calculate_avg_engagement(self, engagement_data: List[StreamingDataPoint]) -> float:
        """Calculate average engagement rate."""
        if not engagement_data:
            return 0.0
        return statistics.mean([dp.value for dp in engagement_data])
    
    def _calculate_interaction_rate(self, interaction_data: List[StreamingDataPoint]) -> float:
        """Calculate interaction rate per minute."""
        if len(interaction_data) < 2:
            return 0.0
        
        time_span = (interaction_data[-1].timestamp - interaction_data[0].timestamp).total_seconds() / 60
        return len(interaction_data) / max(time_span, 1.0)
    
    async def _analyze_user_behavior(self, user_metrics: Dict[str, List[StreamingDataPoint]]) -> List[str]:
        """Analyze user behavior patterns and generate insights."""
        
        insights = []
        
        # Active users analysis
        if user_metrics['active_users']:
            trend = await self._calculate_trend(user_metrics['active_users'])
            current_users = user_metrics['active_users'][-1].value
            
            if trend > 15:
                insights.append(f"🚀 User activity surging with {trend:+.1f}% growth")
            elif trend < -10:
                insights.append(f"⚠️ User activity declining {trend:.1f}%")
            
            if current_users > 100000:
                insights.append(f"🌟 Peak activity period with {current_users:,.0f} concurrent users")
        
        # Engagement analysis
        if user_metrics['engagement']:
            avg_engagement = self._calculate_avg_engagement(user_metrics['engagement'])
            
            if avg_engagement > 80:
                insights.append("💯 Exceptional user engagement levels")
            elif avg_engagement < 40:
                insights.append("📊 User engagement below optimal levels")
        
        # Session behavior
        if user_metrics['sessions']:
            session_values = [dp.value for dp in user_metrics['sessions']]
            avg_session = statistics.mean(session_values)
            
            if avg_session > 45:
                insights.append("🎧 Users highly engaged with long listening sessions")
            elif avg_session < 15:
                insights.append("⚡ Short session lengths indicate quick content consumption")
        
        # Content interaction patterns
        if user_metrics['content_interaction']:
            interaction_rate = self._calculate_interaction_rate(user_metrics['content_interaction'])
            
            if interaction_rate > 50:
                insights.append("🎵 High content interaction rate indicates active discovery")
            elif interaction_rate < 10:
                insights.append("🔍 Low interaction rate suggests passive listening behavior")
        
        return insights


class StreamingMLMetricsFormatter(BaseStreamingFormatter):
    """Formatter for real-time ML model performance metrics."""
    
    async def _format_streaming_data(self, data_points: List[StreamingDataPoint]) -> List[FormattedStreamingData]:
        """Format streaming ML metrics data."""
        
        if not data_points:
            return []
        
        # Categorize ML metrics
        ml_metrics = {
            'model_performance': [],
            'recommendation_engine': [],
            'prediction_accuracy': [],
            'inference_latency': [],
            'model_drift': []
        }
        
        for dp in data_points:
            metric_lower = dp.metric_name.lower()
            if 'accuracy' in metric_lower or 'precision' in metric_lower or 'recall' in metric_lower:
                ml_metrics['model_performance'].append(dp)
            elif 'recommendation' in metric_lower or 'hit_rate' in metric_lower:
                ml_metrics['recommendation_engine'].append(dp)
            elif 'prediction' in metric_lower:
                ml_metrics['prediction_accuracy'].append(dp)
            elif 'latency' in metric_lower or 'inference' in metric_lower:
                ml_metrics['inference_latency'].append(dp)
            elif 'drift' in metric_lower:
                ml_metrics['model_drift'].append(dp)
        
        # Format ML performance dashboard
        ml_dashboard = await self._format_ml_performance_dashboard(ml_metrics)
        
        # Generate ML insights
        ml_insights = await self._analyze_ml_performance(ml_metrics)
        
        # Create alerts if needed
        alerts = await self._generate_ml_alerts(ml_metrics)
        
        timestamp = datetime.now(timezone.utc)
        
        formatted_data = f"""
🤖 **SPOTIFY AI - LIVE ML PERFORMANCE** 🤖

{ml_dashboard}

## 🔍 ML Performance Insights
{chr(10).join(f"• {insight}" for insight in ml_insights)}

{("## 🚨 Active Alerts" + chr(10) + chr(10).join(f"🔴 {alert}" for alert in alerts)) if alerts else "## ✅ All ML Systems Operating Normally"}

*Real-time ML monitoring - Updated: {timestamp.strftime('%H:%M:%S UTC')}*
        """.strip()
        
        # Calculate aggregated ML metrics
        aggregated_ml_metrics = {}
        for category, category_data in ml_metrics.items():
            if category_data:
                aggregated_ml_metrics[category] = await self._calculate_aggregated_metrics(category_data)
        
        metadata = {
            "formatter_type": "streaming_ml_metrics",
            "ml_categories": len([k for k, v in ml_metrics.items() if v]),
            "total_ml_data_points": len(data_points),
            "alerts_count": len(alerts),
            "tenant_id": self.tenant_id,
            "generated_at": timestamp.isoformat()
        }
        
        result = FormattedStreamingData(
            timestamp=timestamp,
            formatted_data=formatted_data,
            raw_data=data_points,
            aggregated_metrics=aggregated_ml_metrics,
            metadata=metadata
        )
        
        return [result]
    
    async def _format_ml_performance_dashboard(self, ml_metrics: Dict[str, List[StreamingDataPoint]]) -> str:
        """Format ML performance dashboard."""
        
        dashboard_sections = []
        
        # Model Performance
        if ml_metrics['model_performance']:
            latest_performance = ml_metrics['model_performance'][-1].value
            perf_trend = await self._calculate_trend(ml_metrics['model_performance'])
            perf_status = "🟢" if latest_performance > 0.9 else "🟡" if latest_performance > 0.8 else "🔴"
            
            dashboard_sections.append(f"""
### 🎯 Model Performance
**Accuracy**: {latest_performance:.3f} {perf_status}
**Trend**: {perf_trend:+.2f}%
            """.strip())
        
        # Recommendation Engine
        if ml_metrics['recommendation_engine']:
            rec_score = ml_metrics['recommendation_engine'][-1].value
            rec_trend = await self._calculate_trend(ml_metrics['recommendation_engine'])
            rec_status = "🟢" if rec_score > 0.7 else "🟡" if rec_score > 0.5 else "🔴"
            
            dashboard_sections.append(f"""
### 🎵 Recommendation Engine
**Hit Rate**: {rec_score:.3f} {rec_status}
**Trend**: {rec_trend:+.2f}%
            """.strip())
        
        # Inference Latency
        if ml_metrics['inference_latency']:
            avg_latency = statistics.mean([dp.value for dp in ml_metrics['inference_latency']])
            latency_trend = await self._calculate_trend(ml_metrics['inference_latency'])
            latency_status = "🟢" if avg_latency < 50 else "🟡" if avg_latency < 100 else "🔴"
            
            dashboard_sections.append(f"""
### ⚡ Inference Performance
**Avg Latency**: {avg_latency:.1f}ms {latency_status}
**Trend**: {latency_trend:+.2f}%
            """.strip())
        
        # Model Drift Detection
        if ml_metrics['model_drift']:
            drift_score = ml_metrics['model_drift'][-1].value
            drift_status = "🟢" if drift_score < 0.1 else "🟡" if drift_score < 0.2 else "🔴"
            
            dashboard_sections.append(f"""
### 🔄 Model Drift
**Drift Score**: {drift_score:.3f} {drift_status}
**Status**: {'Normal' if drift_score < 0.1 else 'Monitoring' if drift_score < 0.2 else 'Action Required'}
            """.strip())
        
        return "\n\n".join(dashboard_sections)
    
    async def _analyze_ml_performance(self, ml_metrics: Dict[str, List[StreamingDataPoint]]) -> List[str]:
        """Analyze ML performance and generate insights."""
        
        insights = []
        
        # Model performance insights
        if ml_metrics['model_performance']:
            latest_perf = ml_metrics['model_performance'][-1].value
            perf_trend = await self._calculate_trend(ml_metrics['model_performance'])
            
            if latest_perf > 0.95:
                insights.append("🌟 Exceptional model performance with high accuracy")
            elif latest_perf < 0.8:
                insights.append("⚠️ Model performance below threshold - consider retraining")
            
            if perf_trend < -5:
                insights.append("📉 Model performance degrading - investigate data quality")
        
        # Recommendation engine insights
        if ml_metrics['recommendation_engine']:
            rec_performance = ml_metrics['recommendation_engine'][-1].value
            
            if rec_performance > 0.8:
                insights.append("🎯 Recommendation engine performing excellently")
            elif rec_performance < 0.5:
                insights.append("🔧 Recommendation hit rate needs optimization")
        
        # Latency insights
        if ml_metrics['inference_latency']:
            latencies = [dp.value for dp in ml_metrics['inference_latency']]
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
            
            if avg_latency < 30:
                insights.append("⚡ Excellent inference latency performance")
            elif p95_latency > 200:
                insights.append("🐌 High latency outliers detected - check infrastructure")
        
        # Model drift insights
        if ml_metrics['model_drift']:
            drift_score = ml_metrics['model_drift'][-1].value
            
            if drift_score > 0.15:
                insights.append("🚨 Significant model drift detected - retrain recommended")
            elif drift_score > 0.1:
                insights.append("👀 Model drift increasing - monitor closely")
        
        return insights
    
    async def _generate_ml_alerts(self, ml_metrics: Dict[str, List[StreamingDataPoint]]) -> List[str]:
        """Generate ML performance alerts."""
        
        alerts = []
        
        # Performance alerts
        if ml_metrics['model_performance']:
            latest_perf = ml_metrics['model_performance'][-1].value
            if latest_perf < 0.75:
                alerts.append(f"Model accuracy critically low: {latest_perf:.3f}")
        
        # Latency alerts
        if ml_metrics['inference_latency']:
            latencies = [dp.value for dp in ml_metrics['inference_latency']]
            avg_latency = statistics.mean(latencies)
            if avg_latency > 150:
                alerts.append(f"High inference latency: {avg_latency:.1f}ms")
        
        # Drift alerts
        if ml_metrics['model_drift']:
            drift_score = ml_metrics['model_drift'][-1].value
            if drift_score > 0.2:
                alerts.append(f"Critical model drift: {drift_score:.3f}")
        
        # Recommendation alerts
        if ml_metrics['recommendation_engine']:
            hit_rate = ml_metrics['recommendation_engine'][-1].value
            if hit_rate < 0.4:
                alerts.append(f"Low recommendation hit rate: {hit_rate:.3f}")
        
        return alerts


# Factory function for creating streaming formatters
def create_streaming_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseStreamingFormatter:
    """
    Factory function to create streaming data formatters.
    
    Args:
        formatter_type: Type of formatter ('real_time', 'user_metrics', 'ml_metrics')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured streaming formatter instance
    """
    formatters = {
        'real_time': RealTimeAnalyticsFormatter,
        'real_time_analytics': RealTimeAnalyticsFormatter,
        'user_metrics': LiveUserMetricsFormatter,
        'live_user_metrics': LiveUserMetricsFormatter,
        'ml_metrics': StreamingMLMetricsFormatter,
        'streaming_ml_metrics': StreamingMLMetricsFormatter,
        'live_analytics': RealTimeAnalyticsFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported streaming formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
