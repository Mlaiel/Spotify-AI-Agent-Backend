"""
Advanced Alert Correlator for Spotify AI Agent
==============================================

Intelligent alert correlation engine that reduces noise and identifies
relationships between different alerts and system events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    tenant_id: str
    metric_name: str
    severity: str
    timestamp: datetime
    value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelatedAlertGroup:
    """Group of correlated alerts."""
    group_id: str
    alerts: List[Alert]
    correlation_score: float
    root_cause_alert: Optional[Alert]
    correlation_type: str
    pattern: str
    timestamp: datetime
    recommendations: List[str]

class AlertCorrelator:
    """
    Advanced alert correlation engine with machine learning capabilities.
    
    Features:
    - Time-based correlation analysis
    - Causal relationship detection
    - Alert clustering and grouping
    - Root cause analysis
    - Pattern recognition
    - Noise reduction through intelligent filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history = deque(maxlen=config.get('max_history', 10000))
        self.correlation_patterns = {}
        self.temporal_window = timedelta(minutes=config.get('temporal_window_minutes', 5))
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
    def correlate_alerts(self, 
                        incoming_alerts: List[Alert]) -> List[CorrelatedAlertGroup]:
        """
        Correlate incoming alerts with historical data.
        
        Args:
            incoming_alerts: New alerts to correlate
            
        Returns:
            List of correlated alert groups
        """
        # Add alerts to history
        self.alert_history.extend(incoming_alerts)
        
        # Get recent alerts for correlation
        recent_alerts = self._get_recent_alerts()
        
        # Perform different types of correlation
        temporal_groups = self._temporal_correlation(recent_alerts)
        causal_groups = self._causal_correlation(recent_alerts)
        pattern_groups = self._pattern_correlation(recent_alerts)
        
        # Merge and deduplicate correlation groups
        all_groups = temporal_groups + causal_groups + pattern_groups
        merged_groups = self._merge_overlapping_groups(all_groups)
        
        # Rank groups by importance
        ranked_groups = self._rank_correlation_groups(merged_groups)
        
        return ranked_groups
        
    def _get_recent_alerts(self) -> List[Alert]:
        """Get alerts within the temporal window."""
        cutoff_time = datetime.now() - self.temporal_window
        return [alert for alert in self.alert_history 
                if alert.timestamp >= cutoff_time]
        
    def _temporal_correlation(self, alerts: List[Alert]) -> List[CorrelatedAlertGroup]:
        """Find alerts that occur close in time."""
        groups = []
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)
        
        # Group alerts by time windows
        time_windows = defaultdict(list)
        for alert in sorted_alerts:
            # Round to nearest minute for grouping
            window_key = alert.timestamp.replace(second=0, microsecond=0)
            time_windows[window_key].append(alert)
            
        # Create correlation groups for windows with multiple alerts
        for window_time, window_alerts in time_windows.items():
            if len(window_alerts) > 1:
                # Calculate correlation score based on alert diversity and timing
                correlation_score = self._calculate_temporal_score(window_alerts)
                
                if correlation_score > 0.5:
                    group = CorrelatedAlertGroup(
                        group_id=f"temporal_{window_time.timestamp()}",
                        alerts=window_alerts,
                        correlation_score=correlation_score,
                        root_cause_alert=self._identify_root_cause(window_alerts),
                        correlation_type="temporal",
                        pattern="concurrent_alerts",
                        timestamp=window_time,
                        recommendations=self._get_temporal_recommendations(window_alerts)
                    )
                    groups.append(group)
                    
        return groups
        
    def _causal_correlation(self, alerts: List[Alert]) -> List[CorrelatedAlertGroup]:
        """Find causal relationships between alerts."""
        groups = []
        
        # Build causality graph
        causality_graph = self._build_causality_graph(alerts)
        
        # Find strongly connected components
        strong_components = list(nx.strongly_connected_components(causality_graph))
        
        for component in strong_components:
            if len(component) > 1:
                component_alerts = [alert for alert in alerts 
                                  if alert.id in component]
                
                # Calculate causal correlation score
                correlation_score = self._calculate_causal_score(
                    component_alerts, causality_graph
                )
                
                if correlation_score > 0.6:
                    group = CorrelatedAlertGroup(
                        group_id=f"causal_{hash(frozenset(component))}",
                        alerts=component_alerts,
                        correlation_score=correlation_score,
                        root_cause_alert=self._find_causal_root(
                            component_alerts, causality_graph
                        ),
                        correlation_type="causal",
                        pattern="cascading_failure",
                        timestamp=min(alert.timestamp for alert in component_alerts),
                        recommendations=self._get_causal_recommendations(component_alerts)
                    )
                    groups.append(group)
                    
        return groups
        
    def _pattern_correlation(self, alerts: List[Alert]) -> List[CorrelatedAlertGroup]:
        """Find alerts matching known patterns."""
        groups = []
        
        # Feature extraction for pattern matching
        alert_features = self._extract_alert_features(alerts)
        
        if len(alert_features) == 0:
            return groups
            
        # Cluster alerts using DBSCAN
        clustering = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(alert_features)
        
        # Group alerts by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                clusters[label].append(alerts[i])
                
        # Create correlation groups for each cluster
        for cluster_id, cluster_alerts in clusters.items():
            if len(cluster_alerts) > 1:
                # Calculate pattern similarity score
                correlation_score = self._calculate_pattern_score(cluster_alerts)
                
                if correlation_score > 0.65:
                    pattern_type = self._identify_pattern_type(cluster_alerts)
                    
                    group = CorrelatedAlertGroup(
                        group_id=f"pattern_{cluster_id}_{datetime.now().timestamp()}",
                        alerts=cluster_alerts,
                        correlation_score=correlation_score,
                        root_cause_alert=self._identify_pattern_root(cluster_alerts),
                        correlation_type="pattern",
                        pattern=pattern_type,
                        timestamp=min(alert.timestamp for alert in cluster_alerts),
                        recommendations=self._get_pattern_recommendations(
                            cluster_alerts, pattern_type
                        )
                    )
                    groups.append(group)
                    
        return groups
        
    def _build_causality_graph(self, alerts: List[Alert]) -> nx.DiGraph:
        """Build a directed graph representing potential causal relationships."""
        graph = nx.DiGraph()
        
        # Add alerts as nodes
        for alert in alerts:
            graph.add_node(alert.id, alert=alert)
            
        # Add edges based on temporal and logical relationships
        for i, alert1 in enumerate(alerts):
            for j, alert2 in enumerate(alerts):
                if i != j:
                    # Check if alert1 could cause alert2
                    if self._could_be_causal(alert1, alert2):
                        causality_strength = self._calculate_causality_strength(
                            alert1, alert2
                        )
                        if causality_strength > 0.5:
                            graph.add_edge(
                                alert1.id, alert2.id,
                                weight=causality_strength
                            )
                            
        return graph
        
    def _could_be_causal(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if alert1 could potentially cause alert2."""
        # Temporal condition: alert1 should occur before alert2
        if alert1.timestamp >= alert2.timestamp:
            return False
            
        # Time window condition: alerts should be close in time
        time_diff = alert2.timestamp - alert1.timestamp
        if time_diff > self.temporal_window:
            return False
            
        # Logical conditions based on metric types
        causal_relationships = {
            'cpu_high': ['response_time_high', 'queue_length_high'],
            'memory_high': ['cpu_high', 'disk_io_high'],
            'disk_full': ['application_error', 'database_error'],
            'network_high': ['response_time_high', 'timeout_errors'],
        }
        
        alert1_type = self._classify_alert_type(alert1)
        alert2_type = self._classify_alert_type(alert2)
        
        if alert1_type in causal_relationships:
            return alert2_type in causal_relationships[alert1_type]
            
        return True  # Default to considering potential causality
        
    def _calculate_causality_strength(self, alert1: Alert, alert2: Alert) -> float:
        """Calculate the strength of causal relationship between two alerts."""
        # Temporal proximity factor
        time_diff = (alert2.timestamp - alert1.timestamp).total_seconds()
        temporal_factor = max(0, 1 - time_diff / self.temporal_window.total_seconds())
        
        # Severity correlation factor
        severity_map = {'critical': 3, 'warning': 2, 'info': 1}
        sev1 = severity_map.get(alert1.severity, 1)
        sev2 = severity_map.get(alert2.severity, 1)
        severity_factor = min(sev1, sev2) / max(sev1, sev2)
        
        # Tenant correlation factor
        tenant_factor = 1.0 if alert1.tenant_id == alert2.tenant_id else 0.5
        
        return temporal_factor * severity_factor * tenant_factor
        
    def _extract_alert_features(self, alerts: List[Alert]) -> np.ndarray:
        """Extract features from alerts for pattern matching."""
        if not alerts:
            return np.array([])
            
        features = []
        
        for alert in alerts:
            # Temporal features
            hour = alert.timestamp.hour
            day_of_week = alert.timestamp.weekday()
            
            # Severity encoding
            severity_map = {'critical': 3, 'warning': 2, 'info': 1}
            severity_score = severity_map.get(alert.severity, 1)
            
            # Metric type encoding
            metric_features = self._encode_metric_type(alert.metric_name)
            
            # Value features
            value_features = [
                alert.value,
                alert.threshold,
                alert.value / alert.threshold if alert.threshold > 0 else 0
            ]
            
            # Combine all features
            alert_features = [
                hour / 24.0,  # Normalize hour
                day_of_week / 6.0,  # Normalize day of week
                severity_score / 3.0,  # Normalize severity
            ] + metric_features + value_features
            
            features.append(alert_features)
            
        return np.array(features)
        
    def _encode_metric_type(self, metric_name: str) -> List[float]:
        """Encode metric type as feature vector."""
        # Common metric categories
        categories = {
            'cpu': [1, 0, 0, 0, 0],
            'memory': [0, 1, 0, 0, 0],
            'disk': [0, 0, 1, 0, 0],
            'network': [0, 0, 0, 1, 0],
            'application': [0, 0, 0, 0, 1]
        }
        
        # Try to match metric name to category
        metric_lower = metric_name.lower()
        for category, encoding in categories.items():
            if category in metric_lower:
                return encoding
                
        # Default encoding for unknown metrics
        return [0, 0, 0, 0, 0]
        
    def _classify_alert_type(self, alert: Alert) -> str:
        """Classify alert into general type category."""
        metric_lower = alert.metric_name.lower()
        
        if 'cpu' in metric_lower:
            return 'cpu_high' if alert.value > alert.threshold else 'cpu_normal'
        elif 'memory' in metric_lower:
            return 'memory_high' if alert.value > alert.threshold else 'memory_normal'
        elif 'disk' in metric_lower:
            return 'disk_full' if alert.value > alert.threshold else 'disk_normal'
        elif 'network' in metric_lower:
            return 'network_high' if alert.value > alert.threshold else 'network_normal'
        elif 'response' in metric_lower or 'latency' in metric_lower:
            return 'response_time_high' if alert.value > alert.threshold else 'response_time_normal'
        else:
            return 'application_error' if alert.severity == 'critical' else 'application_normal'
            
    def _calculate_temporal_score(self, alerts: List[Alert]) -> float:
        """Calculate correlation score for temporally grouped alerts."""
        if len(alerts) < 2:
            return 0.0
            
        # Factor 1: Number of different metrics affected
        unique_metrics = len(set(alert.metric_name for alert in alerts))
        metric_diversity = min(unique_metrics / 5.0, 1.0)  # Cap at 1.0
        
        # Factor 2: Severity distribution
        severities = [alert.severity for alert in alerts]
        critical_count = severities.count('critical')
        warning_count = severities.count('warning')
        severity_score = (critical_count * 3 + warning_count * 2) / (len(alerts) * 3)
        
        # Factor 3: Time span (tighter grouping = higher score)
        time_span = max(alert.timestamp for alert in alerts) - min(alert.timestamp for alert in alerts)
        time_factor = max(0, 1 - time_span.total_seconds() / self.temporal_window.total_seconds())
        
        return (metric_diversity + severity_score + time_factor) / 3.0
        
    def _calculate_causal_score(self, alerts: List[Alert], graph: nx.DiGraph) -> float:
        """Calculate correlation score for causally related alerts."""
        if len(alerts) < 2:
            return 0.0
            
        # Factor 1: Graph connectivity
        alert_ids = [alert.id for alert in alerts]
        subgraph = graph.subgraph(alert_ids)
        connectivity = nx.edge_connectivity(subgraph.to_undirected())
        connectivity_score = min(connectivity / len(alerts), 1.0)
        
        # Factor 2: Causal chain length
        try:
            longest_path = max(len(path) for path in nx.all_simple_paths(
                subgraph, alert_ids[0], alert_ids[-1]
            ))
            chain_score = min(longest_path / len(alerts), 1.0)
        except (nx.NetworkXNoPath, ValueError):
            chain_score = 0.0
            
        # Factor 3: Temporal consistency
        temporal_score = self._calculate_temporal_score(alerts)
        
        return (connectivity_score + chain_score + temporal_score) / 3.0
        
    def _calculate_pattern_score(self, alerts: List[Alert]) -> float:
        """Calculate correlation score for pattern-matched alerts."""
        if len(alerts) < 2:
            return 0.0
            
        # Calculate feature similarity
        features = self._extract_alert_features(alerts)
        if len(features) < 2:
            return 0.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = cosine_similarity([features[i]], [features[j]])[0, 0]
                similarities.append(sim)
                
        return np.mean(similarities) if similarities else 0.0
        
    def _identify_root_cause(self, alerts: List[Alert]) -> Optional[Alert]:
        """Identify the root cause alert in a group."""
        if not alerts:
            return None
            
        # Sort by timestamp to find the earliest alert
        sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)
        
        # Prefer critical alerts as root causes
        critical_alerts = [a for a in sorted_alerts if a.severity == 'critical']
        if critical_alerts:
            return critical_alerts[0]
            
        # Otherwise, return the earliest alert
        return sorted_alerts[0]
        
    def _find_causal_root(self, alerts: List[Alert], graph: nx.DiGraph) -> Optional[Alert]:
        """Find the root cause in a causal chain."""
        if not alerts:
            return None
            
        alert_ids = [alert.id for alert in alerts]
        subgraph = graph.subgraph(alert_ids)
        
        # Find nodes with no incoming edges (potential roots)
        root_candidates = [node for node in subgraph.nodes() 
                          if subgraph.in_degree(node) == 0]
        
        if root_candidates:
            # Return the alert corresponding to the first root candidate
            for alert in alerts:
                if alert.id in root_candidates:
                    return alert
                    
        # Fallback to earliest alert
        return min(alerts, key=lambda a: a.timestamp)
        
    def _identify_pattern_root(self, alerts: List[Alert]) -> Optional[Alert]:
        """Identify the root cause in a pattern-based group."""
        # For pattern-based groups, prefer the most severe alert
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        
        sorted_by_severity = sorted(
            alerts,
            key=lambda a: (severity_order.get(a.severity, 3), a.timestamp)
        )
        
        return sorted_by_severity[0] if sorted_by_severity else None
        
    def _identify_pattern_type(self, alerts: List[Alert]) -> str:
        """Identify the type of pattern in the alert group."""
        # Analyze common characteristics
        metrics = [alert.metric_name for alert in alerts]
        severities = [alert.severity for alert in alerts]
        
        # Check for resource exhaustion pattern
        resource_metrics = ['cpu', 'memory', 'disk']
        if any(any(res in metric.lower() for res in resource_metrics) 
               for metric in metrics):
            return 'resource_exhaustion'
            
        # Check for performance degradation pattern
        perf_metrics = ['response_time', 'latency', 'throughput']
        if any(any(perf in metric.lower() for perf in perf_metrics) 
               for metric in metrics):
            return 'performance_degradation'
            
        # Check for cascading failure pattern
        if len(set(severities)) > 1 and 'critical' in severities:
            return 'cascading_failure'
            
        # Default pattern
        return 'correlated_anomaly'
        
    def _merge_overlapping_groups(self, 
                                 groups: List[CorrelatedAlertGroup]) -> List[CorrelatedAlertGroup]:
        """Merge groups that have overlapping alerts."""
        if not groups:
            return groups
            
        merged = []
        used_groups = set()
        
        for i, group1 in enumerate(groups):
            if i in used_groups:
                continue
                
            merged_alerts = set(alert.id for alert in group1.alerts)
            overlapping_groups = [group1]
            
            for j, group2 in enumerate(groups[i+1:], i+1):
                if j in used_groups:
                    continue
                    
                group2_alerts = set(alert.id for alert in group2.alerts)
                
                # Check for overlap
                if merged_alerts & group2_alerts:
                    overlapping_groups.append(group2)
                    merged_alerts.update(group2_alerts)
                    used_groups.add(j)
                    
            # Create merged group
            all_alerts = []
            for group in overlapping_groups:
                all_alerts.extend(group.alerts)
                
            # Remove duplicates
            unique_alerts = []
            seen_ids = set()
            for alert in all_alerts:
                if alert.id not in seen_ids:
                    unique_alerts.append(alert)
                    seen_ids.add(alert.id)
                    
            # Calculate combined correlation score
            combined_score = np.mean([group.correlation_score for group in overlapping_groups])
            
            merged_group = CorrelatedAlertGroup(
                group_id=f"merged_{group1.group_id}",
                alerts=unique_alerts,
                correlation_score=combined_score,
                root_cause_alert=self._identify_root_cause(unique_alerts),
                correlation_type="merged",
                pattern="mixed_patterns",
                timestamp=min(alert.timestamp for alert in unique_alerts),
                recommendations=self._get_merged_recommendations(overlapping_groups)
            )
            
            merged.append(merged_group)
            used_groups.add(i)
            
        return merged
        
    def _rank_correlation_groups(self, 
                               groups: List[CorrelatedAlertGroup]) -> List[CorrelatedAlertGroup]:
        """Rank correlation groups by importance and urgency."""
        def importance_score(group: CorrelatedAlertGroup) -> float:
            # Factor 1: Correlation strength
            correlation_weight = group.correlation_score * 0.3
            
            # Factor 2: Severity impact
            critical_count = sum(1 for alert in group.alerts if alert.severity == 'critical')
            severity_weight = (critical_count / len(group.alerts)) * 0.4
            
            # Factor 3: Number of affected metrics/tenants
            unique_metrics = len(set(alert.metric_name for alert in group.alerts))
            unique_tenants = len(set(alert.tenant_id for alert in group.alerts))
            diversity_weight = min((unique_metrics + unique_tenants) / 10.0, 1.0) * 0.3
            
            return correlation_weight + severity_weight + diversity_weight
            
        return sorted(groups, key=importance_score, reverse=True)
        
    def _get_temporal_recommendations(self, alerts: List[Alert]) -> List[str]:
        """Get recommendations for temporally correlated alerts."""
        return [
            "Investigate recent system changes or deployments",
            "Check for common infrastructure issues",
            "Review load balancer and network connectivity",
            "Verify database and cache performance"
        ]
        
    def _get_causal_recommendations(self, alerts: List[Alert]) -> List[str]:
        """Get recommendations for causally correlated alerts."""
        return [
            "Address the root cause alert first",
            "Monitor cascade effects during resolution",
            "Implement circuit breakers to prevent propagation",
            "Review system dependencies and coupling"
        ]
        
    def _get_pattern_recommendations(self, alerts: List[Alert], pattern: str) -> List[str]:
        """Get recommendations based on alert pattern."""
        pattern_recommendations = {
            'resource_exhaustion': [
                "Scale resources immediately",
                "Optimize resource-intensive operations",
                "Implement resource quotas and limits"
            ],
            'performance_degradation': [
                "Identify performance bottlenecks",
                "Optimize slow queries and operations",
                "Scale application tier"
            ],
            'cascading_failure': [
                "Implement circuit breakers",
                "Isolate failing components",
                "Gradual service restoration"
            ]
        }
        
        return pattern_recommendations.get(pattern, [
            "Monitor situation closely",
            "Prepare mitigation strategies",
            "Review system architecture"
        ])
        
    def _get_merged_recommendations(self, groups: List[CorrelatedAlertGroup]) -> List[str]:
        """Get recommendations for merged correlation groups."""
        all_recommendations = []
        for group in groups:
            all_recommendations.extend(group.recommendations)
            
        # Remove duplicates and return unique recommendations
        return list(set(all_recommendations))
