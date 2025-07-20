"""
Spotify AI Agent - Advanced Metrics Formatters
=============================================

Ultra-advanced metrics formatting system for multi-platform export,
visualization, and business intelligence integration.

This module handles complex metrics formatting for:
- Prometheus time-series metrics with custom labels
- Grafana dashboard configurations and panels
- InfluxDB optimized time-series data points
- Elasticsearch search-optimized documents
- Custom business intelligence platforms
- Real-time streaming metrics

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

import structlog
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Summary

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Supported metric types with metadata."""
    COUNTER = ("counter", "Cumulative metric that only increases")
    GAUGE = ("gauge", "Metric that can go up and down")
    HISTOGRAM = ("histogram", "Metric with buckets for distribution")
    SUMMARY = ("summary", "Metric with quantiles")
    RATE = ("rate", "Rate of change over time")
    DELTA = ("delta", "Change since last measurement")


class MetricCategory(Enum):
    """Spotify AI specific metric categories."""
    AI_INFERENCE = "ai_inference"
    RECOMMENDATION_ENGINE = "recommendation_engine"
    AUDIO_PROCESSING = "audio_processing"
    USER_ENGAGEMENT = "user_engagement"
    BUSINESS_METRICS = "business_metrics"
    INFRASTRUCTURE = "infrastructure"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COSTS = "costs"
    QUALITY = "quality"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    category: MetricCategory = MetricCategory.INFRASTRUCTURE
    unit: Optional[str] = None
    help_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "type": self.metric_type.value[0],
            "category": self.category.value,
            "unit": self.unit,
            "help": self.help_text
        }


@dataclass
class FormattedMetrics:
    """Container for formatted metrics output."""
    
    content: str
    format_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "format_type": self.format_type,
            "metadata": self.metadata,
            "attachments": self.attachments
        }


class BaseMetricsFormatter(ABC):
    """Abstract base class for all metrics formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
    @abstractmethod
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics for specific platform."""
        pass
        
    def _add_tenant_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Add tenant-specific labels to metrics."""
        enhanced_labels = labels.copy()
        enhanced_labels.update({
            "tenant_id": self.tenant_id,
            "service": "spotify-ai-agent",
            "environment": self.config.get("environment", "production")
        })
        return enhanced_labels
        
    def _validate_metric_name(self, name: str) -> str:
        """Validate and normalize metric name."""
        # Remove invalid characters and normalize
        import re
        normalized = re.sub(r'[^a-zA-Z0-9_:]', '_', name)
        normalized = re.sub(r'_+', '_', normalized)  # Collapse multiple underscores
        normalized = normalized.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it starts with a letter
        if normalized and not normalized[0].isalpha():
            normalized = f"spotify_{normalized}"
            
        return normalized or "unknown_metric"
        
    def _calculate_rate(self, current: MetricPoint, previous: Optional[MetricPoint]) -> Optional[float]:
        """Calculate rate between two metric points."""
        if not previous or current.timestamp <= previous.timestamp:
            return None
            
        time_diff = (current.timestamp - previous.timestamp).total_seconds()
        if time_diff <= 0:
            return None
            
        value_diff = current.value - previous.value
        return value_diff / time_diff


class PrometheusMetricsFormatter(BaseMetricsFormatter):
    """Advanced Prometheus metrics formatter with multi-tenant support."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(tenant_id, config)
        self.registry = CollectorRegistry()
        self.metric_families = {}
        
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics in Prometheus exposition format."""
        
        output_lines = []
        grouped_metrics = self._group_metrics_by_name(metrics)
        
        for metric_name, metric_points in grouped_metrics.items():
            # Validate and normalize metric name
            normalized_name = self._validate_metric_name(metric_name)
            
            # Get metric metadata from first point
            first_point = metric_points[0]
            metric_type = first_point.metric_type
            help_text = first_point.help_text or f"Spotify AI metric: {normalized_name}"
            
            # Add TYPE comment
            output_lines.append(f"# TYPE {normalized_name} {metric_type.value[0]}")
            
            # Add HELP comment
            output_lines.append(f"# HELP {normalized_name} {help_text}")
            
            # Format metric points
            for point in metric_points:
                formatted_line = await self._format_metric_line(normalized_name, point)
                if formatted_line:
                    output_lines.append(formatted_line)
            
            # Add empty line between metrics
            output_lines.append("")
        
        # Add metadata
        metadata = {
            "format_type": "prometheus",
            "metric_count": len(metrics),
            "unique_metrics": len(grouped_metrics),
            "tenant_id": self.tenant_id,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate content size
        content = "\n".join(output_lines)
        metadata["content_size_bytes"] = len(content.encode('utf-8'))
        
        return FormattedMetrics(
            content=content,
            format_type="prometheus",
            metadata=metadata
        )
    
    def _group_metrics_by_name(self, metrics: List[MetricPoint]) -> Dict[str, List[MetricPoint]]:
        """Group metrics by name for proper formatting."""
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric.name].append(metric)
        return dict(grouped)
    
    async def _format_metric_line(self, metric_name: str, point: MetricPoint) -> Optional[str]:
        """Format individual metric line in Prometheus format."""
        try:
            # Add tenant labels
            labels = self._add_tenant_labels(point.labels)
            
            # Format labels
            if labels:
                label_pairs = []
                for key, value in sorted(labels.items()):
                    # Escape label values
                    escaped_value = str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    label_pairs.append(f'{key}="{escaped_value}"')
                
                labels_str = "{" + ",".join(label_pairs) + "}"
            else:
                labels_str = ""
            
            # Format timestamp (Prometheus expects Unix timestamp in milliseconds)
            timestamp_ms = int(point.timestamp.timestamp() * 1000)
            
            # Handle different metric types
            if point.metric_type == MetricType.HISTOGRAM:
                return await self._format_histogram_metric(metric_name, point, labels_str, timestamp_ms)
            elif point.metric_type == MetricType.SUMMARY:
                return await self._format_summary_metric(metric_name, point, labels_str, timestamp_ms)
            else:
                # Standard gauge/counter format
                return f"{metric_name}{labels_str} {point.value} {timestamp_ms}"
                
        except Exception as e:
            self.logger.error("Failed to format metric line", error=str(e), metric=point.name)
            return None
    
    async def _format_histogram_metric(self, metric_name: str, point: MetricPoint, labels_str: str, timestamp_ms: int) -> str:
        """Format histogram metric with buckets."""
        # For histogram, we expect the value to be a dict with buckets
        if not isinstance(point.value, dict) or 'buckets' not in point.value:
            self.logger.warning("Invalid histogram data", metric=metric_name)
            return f"{metric_name}{labels_str} {point.value} {timestamp_ms}"
        
        lines = []
        buckets = point.value['buckets']
        
        # Format bucket metrics
        for le, count in buckets.items():
            bucket_labels = labels_str.rstrip('}')
            if bucket_labels.endswith('{'):
                bucket_labels += f'le="{le}"' + '}'
            else:
                bucket_labels = bucket_labels[:-1] + f',le="{le}"' + '}'
                
            lines.append(f"{metric_name}_bucket{bucket_labels} {count} {timestamp_ms}")
        
        # Add count and sum
        if 'count' in point.value:
            lines.append(f"{metric_name}_count{labels_str} {point.value['count']} {timestamp_ms}")
        if 'sum' in point.value:
            lines.append(f"{metric_name}_sum{labels_str} {point.value['sum']} {timestamp_ms}")
        
        return "\n".join(lines)
    
    async def _format_summary_metric(self, metric_name: str, point: MetricPoint, labels_str: str, timestamp_ms: int) -> str:
        """Format summary metric with quantiles."""
        # For summary, we expect the value to be a dict with quantiles
        if not isinstance(point.value, dict) or 'quantiles' not in point.value:
            self.logger.warning("Invalid summary data", metric=metric_name)
            return f"{metric_name}{labels_str} {point.value} {timestamp_ms}"
        
        lines = []
        quantiles = point.value['quantiles']
        
        # Format quantile metrics
        for quantile, value in quantiles.items():
            quantile_labels = labels_str.rstrip('}')
            if quantile_labels.endswith('{'):
                quantile_labels += f'quantile="{quantile}"' + '}'
            else:
                quantile_labels = quantile_labels[:-1] + f',quantile="{quantile}"' + '}'
                
            lines.append(f"{metric_name}{quantile_labels} {value} {timestamp_ms}")
        
        # Add count and sum
        if 'count' in point.value:
            lines.append(f"{metric_name}_count{labels_str} {point.value['count']} {timestamp_ms}")
        if 'sum' in point.value:
            lines.append(f"{metric_name}_sum{labels_str} {point.value['sum']} {timestamp_ms}")
        
        return "\n".join(lines)
    
    async def format_pushgateway_payload(self, metrics: List[MetricPoint], job_name: str) -> Dict[str, Any]:
        """Format metrics for Prometheus Pushgateway."""
        formatted_metrics = await self.format_metrics(metrics)
        
        return {
            "job": job_name,
            "instance": f"spotify-ai-{self.tenant_id}",
            "metrics": formatted_metrics.content,
            "grouping_key": {
                "tenant_id": self.tenant_id,
                "service": "spotify-ai-agent"
            }
        }


class GrafanaMetricsFormatter(BaseMetricsFormatter):
    """Advanced Grafana dashboard and panel configuration formatter."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(tenant_id, config)
        self.dashboard_templates = self._load_dashboard_templates()
        
    def _load_dashboard_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load Grafana dashboard templates."""
        return {
            "ai_metrics": {
                "title": "Spotify AI - Model Performance",
                "tags": ["spotify", "ai", "ml"],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "5s",
                "panels": []
            },
            "business_metrics": {
                "title": "Spotify AI - Business Intelligence",
                "tags": ["spotify", "business", "kpi"],
                "time": {"from": "now-24h", "to": "now"},
                "refresh": "1m",
                "panels": []
            },
            "infrastructure": {
                "title": "Spotify AI - Infrastructure Monitoring",
                "tags": ["spotify", "infrastructure", "monitoring"],
                "time": {"from": "now-6h", "to": "now"},
                "refresh": "30s",
                "panels": []
            }
        }
    
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics as Grafana data source response."""
        
        # Group metrics by category for appropriate dashboard selection
        categorized_metrics = self._categorize_metrics(metrics)
        
        # Create dashboard configurations
        dashboards = {}
        
        for category, category_metrics in categorized_metrics.items():
            dashboard_config = await self._create_dashboard_config(category, category_metrics)
            dashboards[category.value] = dashboard_config
        
        # Create data source response format
        datasource_response = await self._create_datasource_response(metrics)
        
        content = {
            "dashboards": dashboards,
            "datasource_response": datasource_response,
            "metadata": {
                "tenant_id": self.tenant_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_metrics": len(metrics),
                "categories": list(categorized_metrics.keys())
            }
        }
        
        return FormattedMetrics(
            content=json.dumps(content, indent=2),
            format_type="grafana",
            metadata={
                "dashboard_count": len(dashboards),
                "metric_count": len(metrics)
            }
        )
    
    def _categorize_metrics(self, metrics: List[MetricPoint]) -> Dict[MetricCategory, List[MetricPoint]]:
        """Categorize metrics for appropriate dashboard placement."""
        categorized = defaultdict(list)
        for metric in metrics:
            categorized[metric.category].append(metric)
        return dict(categorized)
    
    async def _create_dashboard_config(self, category: MetricCategory, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Create Grafana dashboard configuration for a category."""
        
        # Get base template
        template_name = self._get_template_name(category)
        dashboard = self.dashboard_templates.get(template_name, self.dashboard_templates["infrastructure"]).copy()
        
        # Update title with tenant info
        dashboard["title"] = f"{dashboard['title']} - {self.tenant_id}"
        
        # Add templating variables
        dashboard["templating"] = {
            "list": [
                {
                    "name": "tenant_id",
                    "type": "constant",
                    "current": {"value": self.tenant_id},
                    "hide": 2
                },
                {
                    "name": "environment",
                    "type": "query",
                    "query": "label_values(environment)",
                    "refresh": 1,
                    "includeAll": True
                }
            ]
        }
        
        # Create panels based on metrics
        panels = []
        panel_id = 1
        
        # Group metrics by type for panel creation
        metric_groups = self._group_metrics_for_panels(metrics)
        
        for group_name, group_metrics in metric_groups.items():
            panel = await self._create_panel(panel_id, group_name, group_metrics, category)
            panels.append(panel)
            panel_id += 1
        
        dashboard["panels"] = panels
        
        # Add annotations
        dashboard["annotations"] = {
            "list": [
                {
                    "name": "Deployments",
                    "datasource": "prometheus",
                    "expr": f"changes(spotify_ai_deployment_timestamp{{tenant_id=\"{self.tenant_id}\"}}[5m]) > 0",
                    "iconColor": "green",
                    "textFormat": "Deployment"
                }
            ]
        }
        
        return dashboard
    
    def _get_template_name(self, category: MetricCategory) -> str:
        """Get dashboard template name based on category."""
        mapping = {
            MetricCategory.AI_INFERENCE: "ai_metrics",
            MetricCategory.RECOMMENDATION_ENGINE: "ai_metrics",
            MetricCategory.AUDIO_PROCESSING: "ai_metrics",
            MetricCategory.USER_ENGAGEMENT: "business_metrics",
            MetricCategory.BUSINESS_METRICS: "business_metrics",
            MetricCategory.COSTS: "business_metrics",
            MetricCategory.INFRASTRUCTURE: "infrastructure",
            MetricCategory.PERFORMANCE: "infrastructure",
            MetricCategory.SECURITY: "infrastructure",
            MetricCategory.QUALITY: "infrastructure"
        }
        return mapping.get(category, "infrastructure")
    
    def _group_metrics_for_panels(self, metrics: List[MetricPoint]) -> Dict[str, List[MetricPoint]]:
        """Group metrics for panel creation."""
        groups = defaultdict(list)
        
        for metric in metrics:
            # Group by common prefixes or patterns
            metric_name = metric.name
            
            if any(keyword in metric_name for keyword in ['latency', 'duration', 'time']):
                groups['Response Times'].append(metric)
            elif any(keyword in metric_name for keyword in ['accuracy', 'precision', 'recall', 'f1']):
                groups['AI Model Performance'].append(metric)
            elif any(keyword in metric_name for keyword in ['throughput', 'rps', 'qps']):
                groups['Throughput'].append(metric)
            elif any(keyword in metric_name for keyword in ['error', 'failure', 'exception']):
                groups['Error Rates'].append(metric)
            elif any(keyword in metric_name for keyword in ['cpu', 'memory', 'disk']):
                groups['Resource Usage'].append(metric)
            elif any(keyword in metric_name for keyword in ['revenue', 'cost', 'price']):
                groups['Financial Metrics'].append(metric)
            elif any(keyword in metric_name for keyword in ['user', 'engagement', 'session']):
                groups['User Metrics'].append(metric)
            else:
                groups['Other Metrics'].append(metric)
        
        return dict(groups)
    
    async def _create_panel(self, panel_id: int, title: str, metrics: List[MetricPoint], category: MetricCategory) -> Dict[str, Any]:
        """Create individual Grafana panel configuration."""
        
        # Determine panel type based on metrics
        panel_type = self._determine_panel_type(metrics)
        
        panel = {
            "id": panel_id,
            "title": title,
            "type": panel_type,
            "gridPos": {"h": 8, "w": 12, "x": (panel_id - 1) % 2 * 12, "y": (panel_id - 1) // 2 * 8},
            "targets": [],
            "options": {},
            "fieldConfig": {
                "defaults": {
                    "unit": self._get_unit_for_metrics(metrics),
                    "decimals": 2
                }
            }
        }
        
        # Create queries for each metric
        for i, metric in enumerate(metrics[:10]):  # Limit to 10 metrics per panel
            query = self._create_prometheus_query(metric)
            target = {
                "expr": query,
                "refId": chr(65 + i),  # A, B, C, etc.
                "legendFormat": self._create_legend_format(metric),
                "interval": "30s"
            }
            panel["targets"].append(target)
        
        # Configure panel options based on type
        if panel_type == "timeseries":
            panel["options"] = {
                "tooltip": {"mode": "multi"},
                "legend": {"displayMode": "table", "values": ["last", "max", "min"]},
                "overrides": []
            }
        elif panel_type == "stat":
            panel["options"] = {
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "background"
            }
        elif panel_type == "gauge":
            panel["options"] = {
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "orientation": "auto",
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
            # Add thresholds for gauge
            panel["fieldConfig"]["defaults"]["thresholds"] = {
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "red", "value": 80}
                ]
            }
        
        return panel
    
    def _determine_panel_type(self, metrics: List[MetricPoint]) -> str:
        """Determine best panel type for metrics."""
        if len(metrics) == 1:
            metric = metrics[0]
            if any(keyword in metric.name for keyword in ['accuracy', 'precision', 'percentage']):
                return "gauge"
            elif metric.metric_type == MetricType.COUNTER:
                return "stat"
            else:
                return "timeseries"
        else:
            return "timeseries"
    
    def _get_unit_for_metrics(self, metrics: List[MetricPoint]) -> str:
        """Determine appropriate unit for metrics."""
        if not metrics:
            return "short"
        
        # Check for common units in metric names
        first_metric = metrics[0]
        name = first_metric.name.lower()
        
        if any(keyword in name for keyword in ['latency', 'duration', 'time']):
            return "ms"
        elif any(keyword in name for keyword in ['bytes', 'memory', 'size']):
            return "bytes"
        elif any(keyword in name for keyword in ['percentage', 'ratio', 'rate']):
            return "percent"
        elif any(keyword in name for keyword in ['rps', 'qps', 'throughput']):
            return "reqps"
        elif any(keyword in name for keyword in ['currency', 'revenue', 'cost']):
            return "currencyUSD"
        else:
            return first_metric.unit or "short"
    
    def _create_prometheus_query(self, metric: MetricPoint) -> str:
        """Create Prometheus query for metric."""
        metric_name = self._validate_metric_name(metric.name)
        
        # Build label selector
        labels = self._add_tenant_labels(metric.labels)
        label_selector = ",".join([f'{k}="{v}"' for k, v in labels.items()])
        
        if label_selector:
            query = f'{metric_name}{{{label_selector}}}'
        else:
            query = metric_name
        
        # Add rate() for counter metrics
        if metric.metric_type == MetricType.COUNTER:
            query = f"rate({query}[5m])"
        
        return query
    
    def _create_legend_format(self, metric: MetricPoint) -> str:
        """Create legend format for metric."""
        # Use metric labels for legend
        if metric.labels:
            # Pick the most relevant labels for legend
            relevant_labels = []
            for label in ['instance', 'job', 'method', 'status', 'endpoint']:
                if label in metric.labels:
                    relevant_labels.append(f"{{{{{label}}}}}")
            
            if relevant_labels:
                return " - ".join(relevant_labels)
        
        return metric.name
    
    async def _create_datasource_response(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Create Grafana datasource response format."""
        
        # Convert metrics to Grafana time series format
        series_data = defaultdict(list)
        
        for metric in metrics:
            series_key = f"{metric.name}_{hash(json.dumps(metric.labels, sort_keys=True))}"
            timestamp_ms = int(metric.timestamp.timestamp() * 1000)
            
            series_data[series_key].append([metric.value, timestamp_ms])
        
        # Format as Grafana expects
        target_data = []
        for series_name, data_points in series_data.items():
            target_data.append({
                "target": series_name,
                "datapoints": sorted(data_points, key=lambda x: x[1])  # Sort by timestamp
            })
        
        return target_data


class InfluxDBMetricsFormatter(BaseMetricsFormatter):
    """Advanced InfluxDB line protocol formatter with optimization."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(tenant_id, config)
        self.precision = config.get('precision', 'ms')  # ns, us, ms, s
        
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics in InfluxDB line protocol."""
        
        lines = []
        
        for metric in metrics:
            line = await self._format_line_protocol(metric)
            if line:
                lines.append(line)
        
        content = "\n".join(lines)
        
        metadata = {
            "format_type": "influxdb",
            "precision": self.precision,
            "line_count": len(lines),
            "content_size": len(content.encode('utf-8')),
            "tenant_id": self.tenant_id
        }
        
        return FormattedMetrics(
            content=content,
            format_type="influxdb", 
            metadata=metadata
        )
    
    async def _format_line_protocol(self, metric: MetricPoint) -> Optional[str]:
        """Format single metric in InfluxDB line protocol."""
        try:
            # Validate measurement name
            measurement = self._validate_metric_name(metric.name)
            
            # Add tenant tags
            tags = self._add_tenant_labels(metric.labels)
            
            # Format tags
            tag_set = []
            for key, value in sorted(tags.items()):
                # Escape tag keys and values
                escaped_key = self._escape_tag_key(key)
                escaped_value = self._escape_tag_value(str(value))
                tag_set.append(f"{escaped_key}={escaped_value}")
            
            tag_string = ",".join(tag_set)
            if tag_string:
                measurement_with_tags = f"{measurement},{tag_string}"
            else:
                measurement_with_tags = measurement
            
            # Format fields
            field_value = self._format_field_value(metric.value)
            fields = f"value={field_value}"
            
            # Add additional fields based on metric type
            if metric.unit:
                fields += f',unit="{metric.unit}"'
            
            if metric.category:
                fields += f',category="{metric.category.value}"'
            
            # Format timestamp
            timestamp = self._format_timestamp(metric.timestamp)
            
            # Combine into line protocol
            line = f"{measurement_with_tags} {fields} {timestamp}"
            
            return line
            
        except Exception as e:
            self.logger.error("Failed to format InfluxDB line", error=str(e), metric=metric.name)
            return None
    
    def _escape_tag_key(self, key: str) -> str:
        """Escape InfluxDB tag key."""
        return key.replace(' ', '\\ ').replace(',', '\\,').replace('=', '\\=')
    
    def _escape_tag_value(self, value: str) -> str:
        """Escape InfluxDB tag value."""
        return value.replace(' ', '\\ ').replace(',', '\\,').replace('=', '\\=')
    
    def _format_field_value(self, value: Union[int, float, str, bool]) -> str:
        """Format field value for InfluxDB."""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, int):
            return f"{value}i"  # Integer notation
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "0.0"  # InfluxDB doesn't accept NaN/Inf
            return str(value)
        elif isinstance(value, str):
            # Escape string values
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        else:
            return f'"{str(value)}"'
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp based on precision."""
        unix_timestamp = timestamp.timestamp()
        
        if self.precision == 'ns':
            return str(int(unix_timestamp * 1_000_000_000))
        elif self.precision == 'us':
            return str(int(unix_timestamp * 1_000_000))
        elif self.precision == 'ms':
            return str(int(unix_timestamp * 1_000))
        else:  # seconds
            return str(int(unix_timestamp))


class ElasticsearchMetricsFormatter(BaseMetricsFormatter):
    """Advanced Elasticsearch document formatter for metrics."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(tenant_id, config)
        self.index_prefix = config.get('index_prefix', 'spotify-ai-metrics')
        
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics as Elasticsearch bulk index operations."""
        
        bulk_operations = []
        
        for metric in metrics:
            # Create index operation
            index_name = self._get_index_name(metric)
            
            index_op = {
                "index": {
                    "_index": index_name,
                    "_type": "_doc",
                    "_id": self._generate_document_id(metric)
                }
            }
            
            # Create document
            document = await self._create_document(metric)
            
            # Add to bulk operations
            bulk_operations.extend([
                json.dumps(index_op),
                json.dumps(document)
            ])
        
        content = "\n".join(bulk_operations) + "\n"  # Elasticsearch requires trailing newline
        
        metadata = {
            "format_type": "elasticsearch",
            "bulk_operations": len(metrics),
            "content_size": len(content.encode('utf-8')),
            "tenant_id": self.tenant_id,
            "indices": list(set(self._get_index_name(m) for m in metrics))
        }
        
        return FormattedMetrics(
            content=content,
            format_type="elasticsearch",
            metadata=metadata
        )
    
    def _get_index_name(self, metric: MetricPoint) -> str:
        """Generate index name based on metric and date."""
        date_suffix = metric.timestamp.strftime("%Y.%m.%d")
        category_suffix = metric.category.value.replace('_', '-')
        
        return f"{self.index_prefix}-{self.tenant_id}-{category_suffix}-{date_suffix}"
    
    def _generate_document_id(self, metric: MetricPoint) -> str:
        """Generate unique document ID."""
        import hashlib
        
        # Create unique ID from metric name, labels, and timestamp
        id_data = f"{metric.name}_{json.dumps(metric.labels, sort_keys=True)}_{metric.timestamp.isoformat()}"
        return hashlib.md5(id_data.encode()).hexdigest()
    
    async def _create_document(self, metric: MetricPoint) -> Dict[str, Any]:
        """Create Elasticsearch document from metric."""
        
        # Add tenant labels
        labels = self._add_tenant_labels(metric.labels)
        
        document = {
            "@timestamp": metric.timestamp.isoformat(),
            "metric": {
                "name": metric.name,
                "value": metric.value,
                "type": metric.metric_type.value[0],
                "category": metric.category.value,
                "unit": metric.unit,
                "help": metric.help_text
            },
            "labels": labels,
            "tenant": {
                "id": self.tenant_id,
                "service": "spotify-ai-agent"
            }
        }
        
        # Add nested fields for better search
        for key, value in labels.items():
            document[f"label_{key}"] = value
        
        # Add time-based fields for easier querying
        document["time"] = {
            "hour": metric.timestamp.hour,
            "day": metric.timestamp.day,
            "month": metric.timestamp.month,
            "year": metric.timestamp.year,
            "weekday": metric.timestamp.weekday()
        }
        
        return document


class CustomMetricsFormatter(BaseMetricsFormatter):
    """Flexible custom metrics formatter with plugin support."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(tenant_id, config)
        self.output_format = config.get('output_format', 'json')
        self.custom_transformers = config.get('transformers', [])
        
    async def format_metrics(self, metrics: List[MetricPoint]) -> FormattedMetrics:
        """Format metrics using custom configuration."""
        
        # Apply custom transformations
        transformed_metrics = metrics
        for transformer in self.custom_transformers:
            transformed_metrics = await self._apply_transformer(transformer, transformed_metrics)
        
        # Format based on output type
        if self.output_format == 'json':
            content = await self._format_as_json(transformed_metrics)
        elif self.output_format == 'csv':
            content = await self._format_as_csv(transformed_metrics)
        elif self.output_format == 'yaml':
            content = await self._format_as_yaml(transformed_metrics)
        else:
            content = await self._format_as_json(transformed_metrics)
        
        metadata = {
            "format_type": self.output_format,
            "metric_count": len(transformed_metrics),
            "transformers_applied": len(self.custom_transformers),
            "tenant_id": self.tenant_id
        }
        
        return FormattedMetrics(
            content=content,
            format_type=self.output_format,
            metadata=metadata
        )
    
    async def _apply_transformer(self, transformer: Dict[str, Any], metrics: List[MetricPoint]) -> List[MetricPoint]:
        """Apply custom transformer to metrics."""
        transformer_type = transformer.get('type')
        
        if transformer_type == 'filter':
            return await self._apply_filter(transformer, metrics)
        elif transformer_type == 'aggregation':
            return await self._apply_aggregation(transformer, metrics)
        elif transformer_type == 'enrichment':
            return await self._apply_enrichment(transformer, metrics)
        else:
            return metrics
    
    async def _apply_filter(self, config: Dict[str, Any], metrics: List[MetricPoint]) -> List[MetricPoint]:
        """Apply filtering transformation."""
        conditions = config.get('conditions', [])
        
        filtered_metrics = []
        for metric in metrics:
            if await self._matches_conditions(metric, conditions):
                filtered_metrics.append(metric)
        
        return filtered_metrics
    
    async def _apply_aggregation(self, config: Dict[str, Any], metrics: List[MetricPoint]) -> List[MetricPoint]:
        """Apply aggregation transformation."""
        group_by = config.get('group_by', [])
        aggregation_func = config.get('function', 'sum')
        
        # Group metrics
        groups = defaultdict(list)
        for metric in metrics:
            group_key = tuple(metric.labels.get(key, '') for key in group_by)
            groups[group_key].append(metric)
        
        # Aggregate each group
        aggregated_metrics = []
        for group_key, group_metrics in groups.items():
            aggregated_metric = await self._aggregate_metrics(group_metrics, aggregation_func)
            aggregated_metrics.append(aggregated_metric)
        
        return aggregated_metrics
    
    async def _apply_enrichment(self, config: Dict[str, Any], metrics: List[MetricPoint]) -> List[MetricPoint]:
        """Apply enrichment transformation."""
        enrichment_data = config.get('data', {})
        
        enriched_metrics = []
        for metric in metrics:
            enriched_metric = MetricPoint(
                name=metric.name,
                value=metric.value,
                timestamp=metric.timestamp,
                labels={**metric.labels, **enrichment_data},
                metric_type=metric.metric_type,
                category=metric.category,
                unit=metric.unit,
                help_text=metric.help_text
            )
            enriched_metrics.append(enriched_metric)
        
        return enriched_metrics
    
    async def _matches_conditions(self, metric: MetricPoint, conditions: List[Dict[str, Any]]) -> bool:
        """Check if metric matches filter conditions."""
        for condition in conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if field == 'name':
                metric_value = metric.name
            elif field == 'category':
                metric_value = metric.category.value
            elif field == 'value':
                metric_value = metric.value
            elif field.startswith('label.'):
                label_key = field[6:]  # Remove 'label.' prefix
                metric_value = metric.labels.get(label_key)
            else:
                continue
            
            if not self._evaluate_condition(metric_value, operator, value):
                return False
        
        return True
    
    def _evaluate_condition(self, metric_value: Any, operator: str, condition_value: Any) -> bool:
        """Evaluate a single condition."""
        if operator == 'equals':
            return metric_value == condition_value
        elif operator == 'not_equals':
            return metric_value != condition_value
        elif operator == 'contains':
            return condition_value in str(metric_value)
        elif operator == 'greater_than':
            return float(metric_value) > float(condition_value)
        elif operator == 'less_than':
            return float(metric_value) < float(condition_value)
        else:
            return True
    
    async def _aggregate_metrics(self, metrics: List[MetricPoint], function: str) -> MetricPoint:
        """Aggregate a group of metrics."""
        if not metrics:
            return None
        
        base_metric = metrics[0]
        values = [m.value for m in metrics]
        
        if function == 'sum':
            aggregated_value = sum(values)
        elif function == 'avg':
            aggregated_value = sum(values) / len(values)
        elif function == 'max':
            aggregated_value = max(values)
        elif function == 'min':
            aggregated_value = min(values)
        elif function == 'count':
            aggregated_value = len(values)
        else:
            aggregated_value = sum(values)  # Default to sum
        
        return MetricPoint(
            name=f"{base_metric.name}_{function}",
            value=aggregated_value,
            timestamp=max(m.timestamp for m in metrics),  # Use latest timestamp
            labels=base_metric.labels,
            metric_type=base_metric.metric_type,
            category=base_metric.category,
            unit=base_metric.unit,
            help_text=f"Aggregated metric ({function}) from {len(metrics)} points"
        )
    
    async def _format_as_json(self, metrics: List[MetricPoint]) -> str:
        """Format metrics as JSON."""
        data = {
            "tenant_id": self.tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": [metric.to_dict() for metric in metrics]
        }
        return json.dumps(data, indent=2)
    
    async def _format_as_csv(self, metrics: List[MetricPoint]) -> str:
        """Format metrics as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ['name', 'value', 'timestamp', 'type', 'category', 'unit', 'tenant_id']
        
        # Add label columns
        all_labels = set()
        for metric in metrics:
            all_labels.update(metric.labels.keys())
        
        label_headers = [f'label_{label}' for label in sorted(all_labels)]
        headers.extend(label_headers)
        
        writer.writerow(headers)
        
        # Write data
        for metric in metrics:
            row = [
                metric.name,
                metric.value,
                metric.timestamp.isoformat(),
                metric.metric_type.value[0],
                metric.category.value,
                metric.unit or '',
                self.tenant_id
            ]
            
            # Add label values
            for label in sorted(all_labels):
                row.append(metric.labels.get(label, ''))
            
            writer.writerow(row)
        
        return output.getvalue()
    
    async def _format_as_yaml(self, metrics: List[MetricPoint]) -> str:
        """Format metrics as YAML."""
        import yaml
        
        data = {
            "tenant_id": self.tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": [metric.to_dict() for metric in metrics]
        }
        
        return yaml.dump(data, default_flow_style=False, indent=2)


# Factory function for creating metrics formatters
def create_metrics_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseMetricsFormatter:
    """
    Factory function to create metrics formatters.
    
    Args:
        formatter_type: Type of formatter ('prometheus', 'grafana', 'influxdb', 'elasticsearch', 'custom')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured formatter instance
    """
    formatters = {
        'prometheus': PrometheusMetricsFormatter,
        'grafana': GrafanaMetricsFormatter,
        'influxdb': InfluxDBMetricsFormatter,
        'elasticsearch': ElasticsearchMetricsFormatter,
        'custom': CustomMetricsFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
