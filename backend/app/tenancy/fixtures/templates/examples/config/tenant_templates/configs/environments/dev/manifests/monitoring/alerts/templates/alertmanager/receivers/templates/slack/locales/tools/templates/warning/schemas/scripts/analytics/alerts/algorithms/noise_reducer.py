"""
Intelligent Noise Reducer for Spotify AI Agent
==============================================

Advanced noise reduction system that filters out false positives and
reduces alert fatigue through intelligent deduplication and suppression.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hashlib
import logging

@dataclass
class NoiseReductionResult:
    """Result of noise reduction analysis."""
    original_alerts: int
    filtered_alerts: int
    noise_reduction_ratio: float
    suppressed_alerts: List[str]
    deduplicated_groups: List[List[str]]
    false_positive_alerts: List[str]
    reason_summary: Dict[str, int]

@dataclass
class AlertSignature:
    """Unique signature for alert identification."""
    metric_name: str
    tenant_id: str
    threshold_range: str
    severity: str
    signature_hash: str

class NoiseReducer:
    """
    Intelligent noise reduction system for alert filtering.
    
    Features:
    - Intelligent deduplication based on similarity
    - Temporal suppression for flapping alerts
    - False positive detection using ML
    - Alert fatigue prevention
    - Context-aware filtering
    - Business hours consideration
    - Maintenance window awareness
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history = deque(maxlen=config.get('max_history', 10000))
        self.suppression_rules = {}
        self.false_positive_patterns = set()
        self.alert_signatures = {}
        self.maintenance_windows = []
        
        # Configuration parameters
        self.deduplication_window = timedelta(minutes=config.get('deduplication_window_minutes', 10))
        self.suppression_threshold = config.get('suppression_threshold', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.flapping_detection_window = timedelta(minutes=config.get('flapping_window_minutes', 30))
        self.max_alerts_per_minute = config.get('max_alerts_per_minute', 20)
        
    def reduce_noise(self, 
                    incoming_alerts: List[Dict[str, Any]],
                    context: Optional[Dict[str, Any]] = None) -> NoiseReductionResult:
        """
        Apply noise reduction to incoming alerts.
        
        Args:
            incoming_alerts: List of alert dictionaries
            context: Additional context for filtering decisions
            
        Returns:
            Noise reduction result with filtered alerts
        """
        if not incoming_alerts:
            return NoiseReductionResult(
                original_alerts=0,
                filtered_alerts=0,
                noise_reduction_ratio=0.0,
                suppressed_alerts=[],
                deduplicated_groups=[],
                false_positive_alerts=[],
                reason_summary={}
            )
            
        original_count = len(incoming_alerts)
        
        # Step 1: Maintenance window filtering
        alerts_after_maintenance = self._filter_maintenance_windows(incoming_alerts)
        
        # Step 2: Rate limiting
        alerts_after_rate_limit = self._apply_rate_limiting(alerts_after_maintenance)
        
        # Step 3: Deduplication
        alerts_after_dedup, dedup_groups = self._deduplicate_alerts(alerts_after_rate_limit)
        
        # Step 4: Flapping detection and suppression
        alerts_after_flapping = self._suppress_flapping_alerts(alerts_after_dedup)
        
        # Step 5: False positive detection
        alerts_after_fp, false_positives = self._filter_false_positives(alerts_after_flapping)
        
        # Step 6: Business logic filtering
        final_alerts = self._apply_business_logic_filtering(alerts_after_fp, context)
        
        # Collect suppression reasons
        suppressed_alerts = []
        reason_summary = defaultdict(int)
        
        # Track which alerts were removed at each step
        self._track_suppression_reasons(
            incoming_alerts, final_alerts, suppressed_alerts, reason_summary
        )
        
        # Update alert history
        self.alert_history.extend(final_alerts)
        
        filtered_count = len(final_alerts)
        noise_reduction_ratio = 1.0 - (filtered_count / original_count) if original_count > 0 else 0.0
        
        return NoiseReductionResult(
            original_alerts=original_count,
            filtered_alerts=filtered_count,
            noise_reduction_ratio=noise_reduction_ratio,
            suppressed_alerts=suppressed_alerts,
            deduplicated_groups=dedup_groups,
            false_positive_alerts=false_positives,
            reason_summary=dict(reason_summary)
        )
        
    def _filter_maintenance_windows(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out alerts that occur during maintenance windows."""
        if not self.maintenance_windows:
            return alerts
            
        filtered_alerts = []
        current_time = datetime.now()
        
        for alert in alerts:
            alert_time = pd.to_datetime(alert.get('timestamp', current_time))
            tenant_id = alert.get('tenant_id', '')
            
            # Check if alert falls within any maintenance window
            is_in_maintenance = False
            for window in self.maintenance_windows:
                if (window.get('tenant_id') == tenant_id and
                    window['start_time'] <= alert_time <= window['end_time']):
                    is_in_maintenance = True
                    break
                    
            if not is_in_maintenance:
                filtered_alerts.append(alert)
            else:
                self.logger.debug(f"Alert {alert.get('id')} suppressed: maintenance window")
                
        return filtered_alerts
        
    def _apply_rate_limiting(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply rate limiting to prevent alert flooding."""
        if not alerts:
            return alerts
            
        # Group alerts by tenant and time windows
        tenant_time_groups = defaultdict(list)
        
        for alert in alerts:
            tenant_id = alert.get('tenant_id', 'unknown')
            alert_time = pd.to_datetime(alert.get('timestamp', datetime.now()))
            
            # Round to nearest minute for grouping
            time_key = alert_time.replace(second=0, microsecond=0)
            group_key = f"{tenant_id}_{time_key}"
            
            tenant_time_groups[group_key].append(alert)
            
        # Apply rate limiting per group
        filtered_alerts = []
        
        for group_key, group_alerts in tenant_time_groups.items():
            if len(group_alerts) <= self.max_alerts_per_minute:
                filtered_alerts.extend(group_alerts)
            else:
                # Keep highest severity alerts up to the limit
                sorted_alerts = sorted(
                    group_alerts,
                    key=lambda a: self._get_severity_priority(a.get('severity', 'info')),
                    reverse=True
                )
                filtered_alerts.extend(sorted_alerts[:self.max_alerts_per_minute])
                
                self.logger.warning(
                    f"Rate limiting applied to {group_key}: "
                    f"{len(group_alerts)} -> {self.max_alerts_per_minute}"
                )
                
        return filtered_alerts
        
    def _get_severity_priority(self, severity: str) -> int:
        """Get numerical priority for severity levels."""
        priority_map = {'critical': 3, 'warning': 2, 'info': 1, 'debug': 0}
        return priority_map.get(severity.lower(), 1)
        
    def _deduplicate_alerts(self, alerts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """Deduplicate similar alerts based on content and timing."""
        if not alerts:
            return alerts, []
            
        # Create alert signatures
        alert_signatures = []
        for alert in alerts:
            signature = self._create_alert_signature(alert)
            alert_signatures.append(signature)
            
        # Group alerts by similarity
        similarity_groups = self._group_similar_alerts(alerts, alert_signatures)
        
        # Keep representative alerts from each group
        deduplicated_alerts = []
        deduplication_groups = []
        
        for group in similarity_groups:
            if len(group) == 1:
                deduplicated_alerts.append(alerts[group[0]])
            else:
                # Choose representative alert (highest severity or most recent)
                representative_idx = self._choose_representative_alert(group, alerts)
                deduplicated_alerts.append(alerts[representative_idx])
                
                # Track deduplication group
                group_ids = [alerts[idx].get('id', f'alert_{idx}') for idx in group]
                deduplication_groups.append(group_ids)
                
                self.logger.debug(f"Deduplicated {len(group)} alerts into 1")
                
        return deduplicated_alerts, deduplication_groups
        
    def _create_alert_signature(self, alert: Dict[str, Any]) -> AlertSignature:
        """Create a unique signature for alert similarity comparison."""
        metric_name = alert.get('metric_name', '')
        tenant_id = alert.get('tenant_id', '')
        severity = alert.get('severity', 'info')
        
        # Create threshold range for grouping similar values
        threshold = float(alert.get('threshold', 0))
        value = float(alert.get('value', 0))
        
        # Group thresholds into ranges for better deduplication
        if threshold > 0:
            threshold_ratio = value / threshold
            if threshold_ratio < 0.8:
                threshold_range = 'low'
            elif threshold_ratio < 1.2:
                threshold_range = 'normal'
            elif threshold_ratio < 2.0:
                threshold_range = 'high'
            else:
                threshold_range = 'critical'
        else:
            threshold_range = 'unknown'
            
        # Create hash for the signature
        signature_data = f"{metric_name}_{tenant_id}_{threshold_range}_{severity}"
        signature_hash = hashlib.md5(signature_data.encode()).hexdigest()[:8]
        
        return AlertSignature(
            metric_name=metric_name,
            tenant_id=tenant_id,
            threshold_range=threshold_range,
            severity=severity,
            signature_hash=signature_hash
        )
        
    def _group_similar_alerts(self, 
                            alerts: List[Dict[str, Any]],
                            signatures: List[AlertSignature]) -> List[List[int]]:
        """Group alerts by similarity using clustering."""
        if len(alerts) <= 1:
            return [[i] for i in range(len(alerts))]
            
        # Create feature matrix for clustering
        features = []
        for i, alert in enumerate(alerts):
            feature_vector = self._extract_similarity_features(alert, signatures[i])
            features.append(feature_vector)
            
        features_array = np.array(features)
        
        # Apply DBSCAN clustering for similarity grouping
        clustering = DBSCAN(
            eps=1.0 - self.similarity_threshold,  # Convert similarity to distance
            min_samples=1,
            metric='euclidean'
        )
        
        try:
            # Scale features for better clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            cluster_labels = clustering.fit_predict(features_scaled)
        except Exception as e:
            self.logger.error(f"Error in similarity clustering: {e}")
            # Fallback: each alert in its own group
            return [[i] for i in range(len(alerts))]
            
        # Group alerts by cluster labels
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
            
        return list(clusters.values())
        
    def _extract_similarity_features(self, 
                                   alert: Dict[str, Any],
                                   signature: AlertSignature) -> List[float]:
        """Extract features for alert similarity comparison."""
        features = []
        
        # Metric type features (one-hot encoding)
        metric_name = alert.get('metric_name', '').lower()
        metric_types = ['cpu', 'memory', 'disk', 'network', 'response', 'error']
        
        for metric_type in metric_types:
            features.append(1.0 if metric_type in metric_name else 0.0)
            
        # Severity encoding
        severity_map = {'critical': 3, 'warning': 2, 'info': 1, 'debug': 0}
        features.append(severity_map.get(alert.get('severity', 'info').lower(), 1))
        
        # Threshold range encoding
        range_map = {'critical': 4, 'high': 3, 'normal': 2, 'low': 1, 'unknown': 0}
        features.append(range_map.get(signature.threshold_range, 0))
        
        # Temporal features
        alert_time = pd.to_datetime(alert.get('timestamp', datetime.now()))
        features.extend([
            alert_time.hour / 24.0,
            alert_time.weekday() / 6.0,
            1.0 if alert_time.weekday() >= 5 else 0.0  # weekend flag
        ])
        
        # Value features (normalized)
        value = float(alert.get('value', 0))
        threshold = float(alert.get('threshold', 1))
        features.extend([
            min(value / max(threshold, 0.1), 10.0),  # threshold ratio (capped)
            min(value / 1000.0, 1.0)  # normalized value (capped)
        ])
        
        return features
        
    def _choose_representative_alert(self, 
                                   group: List[int],
                                   alerts: List[Dict[str, Any]]) -> int:
        """Choose the most representative alert from a group."""
        if len(group) == 1:
            return group[0]
            
        # Scoring criteria for representative selection
        best_score = -1
        best_idx = group[0]
        
        for idx in group:
            alert = alerts[idx]
            score = 0
            
            # Severity score
            severity_scores = {'critical': 4, 'warning': 3, 'info': 2, 'debug': 1}
            score += severity_scores.get(alert.get('severity', 'info').lower(), 1)
            
            # Recency score (more recent = higher score)
            alert_time = pd.to_datetime(alert.get('timestamp', datetime.now()))
            recency_hours = (datetime.now() - alert_time).total_seconds() / 3600
            score += max(0, 5 - recency_hours)  # Up to 5 points for very recent alerts
            
            # Threshold breach severity
            value = float(alert.get('value', 0))
            threshold = float(alert.get('threshold', 1))
            if threshold > 0:
                breach_ratio = value / threshold
                score += min(breach_ratio, 5)  # Up to 5 points for high breaches
                
            if score > best_score:
                best_score = score
                best_idx = idx
                
        return best_idx
        
    def _suppress_flapping_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suppress alerts that are flapping (rapidly changing state)."""
        if not alerts:
            return alerts
            
        # Group alerts by metric and tenant
        metric_groups = defaultdict(list)
        
        for alert in alerts:
            key = f"{alert.get('tenant_id', '')}_{alert.get('metric_name', '')}"
            metric_groups[key].append(alert)
            
        filtered_alerts = []
        
        for group_key, group_alerts in metric_groups.items():
            # Sort by timestamp
            sorted_alerts = sorted(
                group_alerts,
                key=lambda a: pd.to_datetime(a.get('timestamp', datetime.now()))
            )
            
            # Detect flapping pattern
            if self._is_flapping(sorted_alerts):
                # Keep only the latest alert from flapping series
                if sorted_alerts:
                    filtered_alerts.append(sorted_alerts[-1])
                    self.logger.debug(f"Suppressed flapping alerts for {group_key}")
            else:
                filtered_alerts.extend(sorted_alerts)
                
        return filtered_alerts
        
    def _is_flapping(self, sorted_alerts: List[Dict[str, Any]]) -> bool:
        """Detect if alerts represent a flapping condition."""
        if len(sorted_alerts) < 3:
            return False
            
        # Check if alerts alternate between states within the detection window
        now = datetime.now()
        window_start = now - self.flapping_detection_window
        
        recent_alerts = [
            alert for alert in sorted_alerts
            if pd.to_datetime(alert.get('timestamp', now)) >= window_start
        ]
        
        if len(recent_alerts) < 3:
            return False
            
        # Look for alternating severity patterns
        severities = [alert.get('severity', 'info') for alert in recent_alerts]
        
        # Simple flapping detection: check for alternating high/low severity
        alternations = 0
        for i in range(1, len(severities)):
            if severities[i] != severities[i-1]:
                alternations += 1
                
        # Consider flapping if more than 50% of transitions are alternations
        return alternations >= len(severities) * 0.5
        
    def _filter_false_positives(self, alerts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Filter out alerts identified as false positives."""
        if not alerts:
            return alerts, []
            
        filtered_alerts = []
        false_positives = []
        
        for alert in alerts:
            if self._is_false_positive(alert):
                false_positives.append(alert.get('id', ''))
                self.logger.debug(f"Alert {alert.get('id')} identified as false positive")
            else:
                filtered_alerts.append(alert)
                
        return filtered_alerts, false_positives
        
    def _is_false_positive(self, alert: Dict[str, Any]) -> bool:
        """Determine if an alert is likely a false positive."""
        # Check against known false positive patterns
        alert_pattern = self._get_alert_pattern(alert)
        if alert_pattern in self.false_positive_patterns:
            return True
            
        # Statistical analysis for false positive detection
        metric_name = alert.get('metric_name', '')
        tenant_id = alert.get('tenant_id', '')
        value = float(alert.get('value', 0))
        threshold = float(alert.get('threshold', 0))
        
        # Check historical data for this metric
        historical_fps = self._get_historical_false_positives(metric_name, tenant_id)
        
        # If this type of alert has been frequently marked as FP, flag it
        if len(historical_fps) > 5 and threshold > 0:
            similar_fps = [
                fp for fp in historical_fps
                if abs(fp['value'] - value) / threshold < 0.1  # Within 10% of threshold
            ]
            
            if len(similar_fps) >= 3:
                return True
                
        # Business hours consideration for certain metrics
        alert_time = pd.to_datetime(alert.get('timestamp', datetime.now()))
        is_business_hours = 9 <= alert_time.hour <= 17
        
        # Some metrics might be false positives outside business hours
        if not is_business_hours and metric_name.lower() in ['user_sessions', 'api_calls']:
            if value < threshold * 0.5:  # Significantly below threshold
                return True
                
        return False
        
    def _get_alert_pattern(self, alert: Dict[str, Any]) -> str:
        """Get a pattern string for the alert."""
        return f"{alert.get('metric_name', '')}_{alert.get('severity', '')}"
        
    def _get_historical_false_positives(self, 
                                      metric_name: str,
                                      tenant_id: str) -> List[Dict[str, Any]]:
        """Get historical false positives for a metric."""
        # This would typically query a database of historical FP data
        # For now, return empty list
        return []
        
    def _apply_business_logic_filtering(self, 
                                      alerts: List[Dict[str, Any]],
                                      context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply business-specific filtering logic."""
        if not context:
            return alerts
            
        filtered_alerts = []
        
        for alert in alerts:
            # Apply tenant-specific rules
            tenant_id = alert.get('tenant_id', '')
            tenant_config = context.get('tenant_configs', {}).get(tenant_id, {})
            
            # Check if tenant has alert suppression enabled
            if tenant_config.get('suppress_info_alerts', False):
                if alert.get('severity', '').lower() == 'info':
                    continue
                    
            # Check business impact thresholds
            min_business_impact = tenant_config.get('min_business_impact', 0.0)
            alert_impact = alert.get('business_impact', 0.0)
            
            if alert_impact < min_business_impact:
                continue
                
            # Apply custom filtering rules
            custom_rules = tenant_config.get('custom_filter_rules', [])
            if self._passes_custom_rules(alert, custom_rules):
                filtered_alerts.append(alert)
                
        return filtered_alerts
        
    def _passes_custom_rules(self, 
                           alert: Dict[str, Any],
                           rules: List[Dict[str, Any]]) -> bool:
        """Check if alert passes custom filtering rules."""
        for rule in rules:
            rule_type = rule.get('type', '')
            
            if rule_type == 'suppress_metric':
                if alert.get('metric_name', '') == rule.get('metric_name', ''):
                    return False
                    
            elif rule_type == 'severity_threshold':
                min_severity = rule.get('min_severity', 'info')
                alert_severity = alert.get('severity', 'info')
                
                severity_order = ['debug', 'info', 'warning', 'critical']
                if (severity_order.index(alert_severity.lower()) < 
                    severity_order.index(min_severity.lower())):
                    return False
                    
        return True
        
    def _track_suppression_reasons(self, 
                                 original_alerts: List[Dict[str, Any]],
                                 final_alerts: List[Dict[str, Any]],
                                 suppressed_alerts: List[str],
                                 reason_summary: Dict[str, int]):
        """Track reasons for alert suppression."""
        original_ids = {alert.get('id', '') for alert in original_alerts}
        final_ids = {alert.get('id', '') for alert in final_alerts}
        
        suppressed_ids = original_ids - final_ids
        suppressed_alerts.extend(suppressed_ids)
        
        # This is a simplified version - in practice, you'd track
        # the specific reason for each suppression
        if suppressed_ids:
            reason_summary['noise_reduction'] = len(suppressed_ids)
            
    def add_maintenance_window(self, 
                             tenant_id: str,
                             start_time: datetime,
                             end_time: datetime,
                             description: str = ""):
        """Add a maintenance window for alert suppression."""
        self.maintenance_windows.append({
            'tenant_id': tenant_id,
            'start_time': start_time,
            'end_time': end_time,
            'description': description
        })
        
        # Clean up expired maintenance windows
        current_time = datetime.now()
        self.maintenance_windows = [
            window for window in self.maintenance_windows
            if window['end_time'] > current_time
        ]
        
    def add_false_positive_pattern(self, pattern: str):
        """Add a pattern to the false positive detection list."""
        self.false_positive_patterns.add(pattern)
        
    def get_noise_reduction_stats(self) -> Dict[str, Any]:
        """Get noise reduction statistics."""
        total_processed = len(self.alert_history)
        
        return {
            'total_alerts_processed': total_processed,
            'active_maintenance_windows': len(self.maintenance_windows),
            'false_positive_patterns': len(self.false_positive_patterns),
            'configuration': {
                'deduplication_window_minutes': self.deduplication_window.total_seconds() / 60,
                'similarity_threshold': self.similarity_threshold,
                'max_alerts_per_minute': self.max_alerts_per_minute,
                'suppression_threshold': self.suppression_threshold
            }
        }
