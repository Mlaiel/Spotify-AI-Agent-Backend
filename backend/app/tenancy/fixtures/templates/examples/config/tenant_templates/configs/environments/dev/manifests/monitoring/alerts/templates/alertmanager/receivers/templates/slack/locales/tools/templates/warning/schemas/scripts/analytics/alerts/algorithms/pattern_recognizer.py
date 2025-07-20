"""
Pattern Recognition Engine for Spotify AI Agent
===============================================

Advanced pattern recognition system for identifying recurring alert patterns,
behavioral anomalies, and predictive insights from alert data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
import logging

@dataclass
class AlertPattern:
    """Identified alert pattern with metadata."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    description: str
    affected_metrics: List[str]
    typical_duration: timedelta
    severity_distribution: Dict[str, int]
    temporal_characteristics: Dict[str, Any]
    recommendations: List[str]
    examples: List[str]

@dataclass
class PatternPrediction:
    """Prediction of pattern occurrence."""
    pattern_id: str
    predicted_occurrence: datetime
    confidence: float
    contributing_factors: List[str]
    preventive_actions: List[str]

class PatternRecognizer:
    """
    Machine learning-based pattern recognition for alert analysis.
    
    Features:
    - Temporal pattern detection
    - Sequence pattern mining
    - Anomaly pattern identification
    - Behavioral clustering
    - Predictive pattern analysis
    - Multi-dimensional pattern correlation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.known_patterns = {}
        self.pattern_models = {}
        self.sequence_patterns = {}
        self.temporal_patterns = {}
        self.min_pattern_occurrences = config.get('min_pattern_occurrences', 3)
        self.pattern_window = timedelta(hours=config.get('pattern_window_hours', 24))
        
    def analyze_patterns(self, 
                        alert_history: List[Dict[str, Any]],
                        tenant_id: str) -> List[AlertPattern]:
        """
        Analyze alert history to identify patterns.
        
        Args:
            alert_history: Historical alert data
            tenant_id: Tenant identifier
            
        Returns:
            List of identified alert patterns
        """
        if not alert_history:
            return []
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(alert_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Different pattern analysis approaches
        temporal_patterns = self._analyze_temporal_patterns(df, tenant_id)
        sequence_patterns = self._analyze_sequence_patterns(df, tenant_id)
        frequency_patterns = self._analyze_frequency_patterns(df, tenant_id)
        correlation_patterns = self._analyze_correlation_patterns(df, tenant_id)
        
        # Combine and deduplicate patterns
        all_patterns = (temporal_patterns + sequence_patterns + 
                       frequency_patterns + correlation_patterns)
        
        # Filter and rank patterns
        significant_patterns = self._filter_significant_patterns(all_patterns)
        ranked_patterns = self._rank_patterns(significant_patterns)
        
        return ranked_patterns
        
    def _analyze_temporal_patterns(self, 
                                 df: pd.DataFrame,
                                 tenant_id: str) -> List[AlertPattern]:
        """Analyze temporal patterns in alerts."""
        patterns = []
        
        # Add temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Analyze hourly patterns
        hourly_patterns = self._find_hourly_patterns(df, tenant_id)
        patterns.extend(hourly_patterns)
        
        # Analyze daily patterns
        daily_patterns = self._find_daily_patterns(df, tenant_id)
        patterns.extend(daily_patterns)
        
        # Analyze seasonal patterns
        seasonal_patterns = self._find_seasonal_patterns(df, tenant_id)
        patterns.extend(seasonal_patterns)
        
        return patterns
        
    def _find_hourly_patterns(self, 
                            df: pd.DataFrame,
                            tenant_id: str) -> List[AlertPattern]:
        """Find patterns based on hour of day."""
        patterns = []
        
        # Group by hour and analyze alert frequency
        hourly_counts = df.groupby(['hour', 'metric_name']).size().reset_index(name='count')
        
        # Find hours with significantly high alert counts
        for metric in df['metric_name'].unique():
            metric_hourly = hourly_counts[hourly_counts['metric_name'] == metric]
            
            if len(metric_hourly) < 3:
                continue
                
            mean_count = metric_hourly['count'].mean()
            std_count = metric_hourly['count'].std()
            threshold = mean_count + 2 * std_count
            
            high_activity_hours = metric_hourly[metric_hourly['count'] > threshold]
            
            if len(high_activity_hours) > 0:
                pattern = AlertPattern(
                    pattern_id=f"hourly_{tenant_id}_{metric}_{hash(str(high_activity_hours['hour'].tolist()))}",
                    pattern_type="temporal_hourly",
                    frequency=high_activity_hours['count'].sum(),
                    confidence=min(1.0, len(high_activity_hours) / 24.0 * 2),
                    description=f"High alert activity for {metric} during hours: {high_activity_hours['hour'].tolist()}",
                    affected_metrics=[metric],
                    typical_duration=timedelta(hours=len(high_activity_hours)),
                    severity_distribution=self._get_severity_distribution(
                        df[(df['metric_name'] == metric) & 
                           (df['hour'].isin(high_activity_hours['hour']))]
                    ),
                    temporal_characteristics={
                        'peak_hours': high_activity_hours['hour'].tolist(),
                        'peak_intensity': high_activity_hours['count'].max(),
                        'pattern_type': 'hourly_recurring'
                    },
                    recommendations=[
                        f"Investigate system behavior during peak hours: {high_activity_hours['hour'].tolist()}",
                        "Consider proactive scaling during these periods",
                        "Review batch jobs or scheduled tasks"
                    ],
                    examples=self._get_pattern_examples(
                        df[(df['metric_name'] == metric) & 
                           (df['hour'].isin(high_activity_hours['hour']))], 3
                    )
                )
                patterns.append(pattern)
                
        return patterns
        
    def _find_daily_patterns(self, 
                           df: pd.DataFrame,
                           tenant_id: str) -> List[AlertPattern]:
        """Find patterns based on day of week."""
        patterns = []
        
        # Group by day of week and analyze patterns
        daily_counts = df.groupby(['day_of_week', 'metric_name']).size().reset_index(name='count')
        
        for metric in df['metric_name'].unique():
            metric_daily = daily_counts[daily_counts['metric_name'] == metric]
            
            if len(metric_daily) < 3:
                continue
                
            # Check for weekend vs weekday patterns
            weekdays = metric_daily[metric_daily['day_of_week'] < 5]['count'].mean()
            weekends = metric_daily[metric_daily['day_of_week'] >= 5]['count'].mean()
            
            if weekends > 0 and weekdays > 0:
                ratio = max(weekends, weekdays) / min(weekends, weekdays)
                
                if ratio > 2.0:  # Significant difference
                    pattern_type = "weekend_high" if weekends > weekdays else "weekday_high"
                    
                    pattern = AlertPattern(
                        pattern_id=f"daily_{tenant_id}_{metric}_{pattern_type}",
                        pattern_type="temporal_daily",
                        frequency=int(metric_daily['count'].sum()),
                        confidence=min(1.0, ratio / 5.0),
                        description=f"{metric} shows higher alert activity on {'weekends' if weekends > weekdays else 'weekdays'}",
                        affected_metrics=[metric],
                        typical_duration=timedelta(days=2 if weekends > weekdays else 5),
                        severity_distribution=self._get_severity_distribution(
                            df[df['metric_name'] == metric]
                        ),
                        temporal_characteristics={
                            'weekday_avg': weekdays,
                            'weekend_avg': weekends,
                            'ratio': ratio,
                            'pattern_type': pattern_type
                        },
                        recommendations=[
                            f"Investigate {'weekend' if weekends > weekdays else 'weekday'} specific issues",
                            "Review user activity patterns",
                            "Adjust monitoring thresholds for different days"
                        ],
                        examples=self._get_pattern_examples(df[df['metric_name'] == metric], 3)
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _find_seasonal_patterns(self, 
                              df: pd.DataFrame,
                              tenant_id: str) -> List[AlertPattern]:
        """Find seasonal/monthly patterns."""
        patterns = []
        
        if len(df['month'].unique()) < 3:
            return patterns  # Need at least 3 months of data
            
        monthly_counts = df.groupby(['month', 'metric_name']).size().reset_index(name='count')
        
        for metric in df['metric_name'].unique():
            metric_monthly = monthly_counts[monthly_counts['metric_name'] == metric]
            
            if len(metric_monthly) < 3:
                continue
                
            # Find months with unusually high activity
            mean_count = metric_monthly['count'].mean()
            std_count = metric_monthly['count'].std()
            
            if std_count > 0:
                high_months = metric_monthly[
                    metric_monthly['count'] > mean_count + std_count
                ]
                
                if len(high_months) > 0:
                    pattern = AlertPattern(
                        pattern_id=f"seasonal_{tenant_id}_{metric}_{hash(str(high_months['month'].tolist()))}",
                        pattern_type="temporal_seasonal",
                        frequency=high_months['count'].sum(),
                        confidence=min(1.0, len(high_months) / 12.0 * 3),
                        description=f"{metric} shows seasonal patterns in months: {high_months['month'].tolist()}",
                        affected_metrics=[metric],
                        typical_duration=timedelta(days=30 * len(high_months)),
                        severity_distribution=self._get_severity_distribution(
                            df[df['metric_name'] == metric]
                        ),
                        temporal_characteristics={
                            'high_activity_months': high_months['month'].tolist(),
                            'seasonal_intensity': high_months['count'].max(),
                            'pattern_type': 'seasonal_recurring'
                        },
                        recommendations=[
                            "Prepare for seasonal load variations",
                            "Plan capacity scaling for high-activity months",
                            "Review business cycles and seasonal trends"
                        ],
                        examples=self._get_pattern_examples(df[df['metric_name'] == metric], 3)
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _analyze_sequence_patterns(self, 
                                 df: pd.DataFrame,
                                 tenant_id: str) -> List[AlertPattern]:
        """Analyze sequential patterns in alerts."""
        patterns = []
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Create sequences of consecutive alerts
        sequences = self._extract_alert_sequences(df_sorted)
        
        # Find frequent sequences
        frequent_sequences = self._find_frequent_sequences(sequences)
        
        for sequence, frequency in frequent_sequences.items():
            if frequency >= self.min_pattern_occurrences:
                pattern = AlertPattern(
                    pattern_id=f"sequence_{tenant_id}_{hash(sequence)}",
                    pattern_type="sequential",
                    frequency=frequency,
                    confidence=min(1.0, frequency / len(sequences) * 3),
                    description=f"Sequential pattern: {' → '.join(sequence)}",
                    affected_metrics=list(set(sequence)),
                    typical_duration=self._estimate_sequence_duration(df_sorted, sequence),
                    severity_distribution=self._get_sequence_severity_distribution(df_sorted, sequence),
                    temporal_characteristics={
                        'sequence_length': len(sequence),
                        'sequence_pattern': sequence,
                        'average_interval': self._calculate_sequence_intervals(df_sorted, sequence)
                    },
                    recommendations=[
                        "Investigate causal relationships in sequence",
                        "Consider implementing circuit breakers",
                        "Review system dependencies"
                    ],
                    examples=self._get_sequence_examples(df_sorted, sequence, 3)
                )
                patterns.append(pattern)
                
        return patterns
        
    def _analyze_frequency_patterns(self, 
                                  df: pd.DataFrame,
                                  tenant_id: str) -> List[AlertPattern]:
        """Analyze frequency-based patterns."""
        patterns = []
        
        # Analyze alert frequency for each metric
        for metric in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric].copy()
            
            if len(metric_data) < self.min_pattern_occurrences:
                continue
                
            # Calculate time intervals between alerts
            metric_data = metric_data.sort_values('timestamp')
            intervals = metric_data['timestamp'].diff().dt.total_seconds().dropna()
            
            if len(intervals) < 2:
                continue
                
            # Analyze interval patterns
            mean_interval = intervals.mean()
            std_interval = intervals.std()
            
            # Check for regular intervals (potential recurring pattern)
            if std_interval < mean_interval * 0.3:  # Low variance indicates regularity
                pattern = AlertPattern(
                    pattern_id=f"frequency_{tenant_id}_{metric}",
                    pattern_type="frequency_regular",
                    frequency=len(metric_data),
                    confidence=max(0.1, 1.0 - (std_interval / mean_interval)),
                    description=f"{metric} alerts occur at regular intervals (~{mean_interval/3600:.1f} hours)",
                    affected_metrics=[metric],
                    typical_duration=timedelta(seconds=mean_interval),
                    severity_distribution=self._get_severity_distribution(metric_data),
                    temporal_characteristics={
                        'mean_interval_hours': mean_interval / 3600,
                        'interval_std_hours': std_interval / 3600,
                        'regularity_score': 1.0 - (std_interval / mean_interval),
                        'pattern_type': 'regular_frequency'
                    },
                    recommendations=[
                        "Investigate recurring system behavior",
                        "Check for scheduled tasks or batch jobs",
                        "Consider threshold adjustments if false positives"
                    ],
                    examples=self._get_pattern_examples(metric_data, 3)
                )
                patterns.append(pattern)
                
        return patterns
        
    def _analyze_correlation_patterns(self, 
                                    df: pd.DataFrame,
                                    tenant_id: str) -> List[AlertPattern]:
        """Analyze correlation patterns between different metrics."""
        patterns = []
        
        # Create metric co-occurrence matrix
        metric_combinations = self._find_metric_correlations(df)
        
        for (metric1, metric2), correlation_data in metric_combinations.items():
            if correlation_data['co_occurrence'] >= self.min_pattern_occurrences:
                pattern = AlertPattern(
                    pattern_id=f"correlation_{tenant_id}_{metric1}_{metric2}",
                    pattern_type="correlation",
                    frequency=correlation_data['co_occurrence'],
                    confidence=correlation_data['correlation_strength'],
                    description=f"Strong correlation between {metric1} and {metric2} alerts",
                    affected_metrics=[metric1, metric2],
                    typical_duration=correlation_data['average_duration'],
                    severity_distribution=correlation_data['severity_distribution'],
                    temporal_characteristics={
                        'correlation_strength': correlation_data['correlation_strength'],
                        'average_time_gap': correlation_data['average_time_gap'],
                        'pattern_type': 'metric_correlation'
                    },
                    recommendations=[
                        "Investigate common root causes",
                        "Consider consolidating related alerts",
                        "Review system architecture dependencies"
                    ],
                    examples=correlation_data['examples']
                )
                patterns.append(pattern)
                
        return patterns
        
    def _extract_alert_sequences(self, df: pd.DataFrame) -> List[Tuple[str, ...]]:
        """Extract sequences of consecutive alerts."""
        sequences = []
        
        # Group alerts into windows
        window_size = self.pattern_window
        
        for i in range(len(df)):
            window_start = df.iloc[i]['timestamp']
            window_end = window_start + window_size
            
            window_alerts = df[
                (df['timestamp'] >= window_start) & 
                (df['timestamp'] <= window_end)
            ]['metric_name'].tolist()
            
            if len(window_alerts) > 1:
                sequences.append(tuple(window_alerts))
                
        return sequences
        
    def _find_frequent_sequences(self, sequences: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
        """Find frequently occurring sequences."""
        sequence_counts = Counter(sequences)
        
        # Filter by minimum frequency
        frequent = {seq: count for seq, count in sequence_counts.items() 
                   if count >= self.min_pattern_occurrences}
                   
        return frequent
        
    def _find_metric_correlations(self, df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Find correlations between different metrics."""
        correlations = {}
        
        metrics = df['metric_name'].unique()
        
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                # Find co-occurring alerts
                metric1_times = set(df[df['metric_name'] == metric1]['timestamp'])
                metric2_times = set(df[df['metric_name'] == metric2]['timestamp'])
                
                # Find alerts that occur within the correlation window
                co_occurrences = 0
                time_gaps = []
                
                for t1 in metric1_times:
                    for t2 in metric2_times:
                        gap = abs((t2 - t1).total_seconds())
                        if gap <= self.pattern_window.total_seconds():
                            co_occurrences += 1
                            time_gaps.append(gap)
                            
                if co_occurrences > 0:
                    # Calculate correlation strength
                    correlation_strength = min(1.0, co_occurrences / max(len(metric1_times), len(metric2_times)))
                    
                    correlations[(metric1, metric2)] = {
                        'co_occurrence': co_occurrences,
                        'correlation_strength': correlation_strength,
                        'average_time_gap': np.mean(time_gaps) if time_gaps else 0,
                        'average_duration': timedelta(seconds=np.mean(time_gaps)) if time_gaps else timedelta(0),
                        'severity_distribution': self._get_correlation_severity_distribution(df, metric1, metric2),
                        'examples': self._get_correlation_examples(df, metric1, metric2, 3)
                    }
                    
        return correlations
        
    def predict_pattern_occurrence(self, 
                                 patterns: List[AlertPattern],
                                 current_context: Dict[str, Any]) -> List[PatternPrediction]:
        """Predict when patterns might occur next."""
        predictions = []
        
        for pattern in patterns:
            if pattern.pattern_type == "temporal_hourly":
                prediction = self._predict_hourly_pattern(pattern, current_context)
            elif pattern.pattern_type == "temporal_daily":
                prediction = self._predict_daily_pattern(pattern, current_context)
            elif pattern.pattern_type == "temporal_seasonal":
                prediction = self._predict_seasonal_pattern(pattern, current_context)
            elif pattern.pattern_type == "frequency_regular":
                prediction = self._predict_frequency_pattern(pattern, current_context)
            else:
                continue
                
            if prediction:
                predictions.append(prediction)
                
        return sorted(predictions, key=lambda p: p.predicted_occurrence)
        
    def _predict_hourly_pattern(self, 
                              pattern: AlertPattern,
                              context: Dict[str, Any]) -> Optional[PatternPrediction]:
        """Predict next occurrence of hourly pattern."""
        peak_hours = pattern.temporal_characteristics.get('peak_hours', [])
        if not peak_hours:
            return None
            
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Find next peak hour
        next_peak_hours = [h for h in peak_hours if h > current_hour]
        if next_peak_hours:
            next_hour = min(next_peak_hours)
            next_occurrence = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        else:
            # Next occurrence is tomorrow
            next_hour = min(peak_hours)
            next_occurrence = (current_time + timedelta(days=1)).replace(
                hour=next_hour, minute=0, second=0, microsecond=0
            )
            
        return PatternPrediction(
            pattern_id=pattern.pattern_id,
            predicted_occurrence=next_occurrence,
            confidence=pattern.confidence * 0.8,  # Reduce confidence for prediction
            contributing_factors=[
                f"Historical peak activity at hour {next_hour}",
                f"Pattern confidence: {pattern.confidence:.2f}"
            ],
            preventive_actions=[
                "Monitor system resources closely",
                "Prepare scaling procedures",
                "Review recent system changes"
            ]
        )
        
    def _predict_daily_pattern(self, 
                             pattern: AlertPattern,
                             context: Dict[str, Any]) -> Optional[PatternPrediction]:
        """Predict next occurrence of daily pattern."""
        # Implementation similar to hourly but for days of week
        current_time = datetime.now()
        pattern_type = pattern.temporal_characteristics.get('pattern_type')
        
        if pattern_type == "weekend_high":
            # Next weekend
            days_until_weekend = (5 - current_time.weekday()) % 7
            if days_until_weekend == 0 and current_time.weekday() >= 5:
                days_until_weekend = 7  # Next weekend
            next_occurrence = current_time + timedelta(days=days_until_weekend)
        elif pattern_type == "weekday_high":
            # Next weekday
            if current_time.weekday() >= 5:  # Weekend
                days_until_weekday = 7 - current_time.weekday()
            else:
                days_until_weekday = 1  # Tomorrow if it's a weekday
            next_occurrence = current_time + timedelta(days=days_until_weekday)
        else:
            return None
            
        return PatternPrediction(
            pattern_id=pattern.pattern_id,
            predicted_occurrence=next_occurrence,
            confidence=pattern.confidence * 0.7,
            contributing_factors=[
                f"Historical {pattern_type} pattern",
                f"Pattern occurs on {'weekends' if pattern_type == 'weekend_high' else 'weekdays'}"
            ],
            preventive_actions=[
                "Adjust monitoring sensitivity",
                "Prepare appropriate staffing",
                "Review system capacity"
            ]
        )
        
    def _predict_seasonal_pattern(self, 
                                pattern: AlertPattern,
                                context: Dict[str, Any]) -> Optional[PatternPrediction]:
        """Predict next occurrence of seasonal pattern."""
        high_months = pattern.temporal_characteristics.get('high_activity_months', [])
        if not high_months:
            return None
            
        current_time = datetime.now()
        current_month = current_time.month
        
        # Find next high activity month
        next_months = [m for m in high_months if m > current_month]
        if next_months:
            next_month = min(next_months)
            next_occurrence = current_time.replace(month=next_month, day=1)
        else:
            # Next year
            next_month = min(high_months)
            next_occurrence = current_time.replace(year=current_time.year + 1, month=next_month, day=1)
            
        return PatternPrediction(
            pattern_id=pattern.pattern_id,
            predicted_occurrence=next_occurrence,
            confidence=pattern.confidence * 0.6,
            contributing_factors=[
                f"Seasonal pattern in month {next_month}",
                "Historical seasonal activity"
            ],
            preventive_actions=[
                "Plan seasonal capacity scaling",
                "Review historical incidents",
                "Prepare operational procedures"
            ]
        )
        
    def _predict_frequency_pattern(self, 
                                 pattern: AlertPattern,
                                 context: Dict[str, Any]) -> Optional[PatternPrediction]:
        """Predict next occurrence of frequency-based pattern."""
        mean_interval = pattern.temporal_characteristics.get('mean_interval_hours', 0)
        if mean_interval <= 0:
            return None
            
        # Estimate next occurrence based on last known occurrence
        current_time = datetime.now()
        next_occurrence = current_time + timedelta(hours=mean_interval)
        
        return PatternPrediction(
            pattern_id=pattern.pattern_id,
            predicted_occurrence=next_occurrence,
            confidence=pattern.confidence * 0.9,
            contributing_factors=[
                f"Regular interval pattern: {mean_interval:.1f} hours",
                f"High regularity score: {pattern.temporal_characteristics.get('regularity_score', 0):.2f}"
            ],
            preventive_actions=[
                "Monitor for pattern deviation",
                "Investigate underlying cause",
                "Consider threshold adjustment"
            ]
        )
        
    def _filter_significant_patterns(self, patterns: List[AlertPattern]) -> List[AlertPattern]:
        """Filter patterns to keep only significant ones."""
        # Remove patterns with low confidence
        min_confidence = self.config.get('min_pattern_confidence', 0.3)
        significant = [p for p in patterns if p.confidence >= min_confidence]
        
        # Remove duplicate patterns (same affected metrics and similar characteristics)
        deduplicated = []
        for pattern in significant:
            is_duplicate = False
            for existing in deduplicated:
                if self._patterns_are_similar(pattern, existing):
                    # Keep the one with higher confidence
                    if pattern.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(pattern)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                deduplicated.append(pattern)
                
        return deduplicated
        
    def _patterns_are_similar(self, pattern1: AlertPattern, pattern2: AlertPattern) -> bool:
        """Check if two patterns are similar enough to be considered duplicates."""
        # Same affected metrics
        if set(pattern1.affected_metrics) == set(pattern2.affected_metrics):
            # Similar pattern types
            if pattern1.pattern_type == pattern2.pattern_type:
                return True
                
        return False
        
    def _rank_patterns(self, patterns: List[AlertPattern]) -> List[AlertPattern]:
        """Rank patterns by importance and actionability."""
        def pattern_score(pattern: AlertPattern) -> float:
            # Confidence weight
            confidence_score = pattern.confidence * 0.4
            
            # Frequency weight (more frequent = more important)
            max_frequency = max(p.frequency for p in patterns) if patterns else 1
            frequency_score = (pattern.frequency / max_frequency) * 0.3
            
            # Affected metrics weight (more metrics = more important)
            metrics_score = min(len(pattern.affected_metrics) / 5.0, 1.0) * 0.2
            
            # Severity weight (critical alerts = more important)
            critical_ratio = pattern.severity_distribution.get('critical', 0) / max(sum(pattern.severity_distribution.values()), 1)
            severity_score = critical_ratio * 0.1
            
            return confidence_score + frequency_score + metrics_score + severity_score
            
        return sorted(patterns, key=pattern_score, reverse=True)
        
    # Helper methods for getting pattern metadata
    def _get_severity_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get severity distribution for a subset of alerts."""
        if 'severity' not in df.columns:
            return {}
        return df['severity'].value_counts().to_dict()
        
    def _get_pattern_examples(self, df: pd.DataFrame, limit: int = 3) -> List[str]:
        """Get example alerts for a pattern."""
        examples = []
        for _, row in df.head(limit).iterrows():
            example = f"{row['timestamp']}: {row['metric_name']} = {row.get('value', 'N/A')} ({row.get('severity', 'unknown')})"
            examples.append(example)
        return examples
        
    def _estimate_sequence_duration(self, df: pd.DataFrame, sequence: Tuple[str, ...]) -> timedelta:
        """Estimate typical duration of a sequence pattern."""
        # This is a simplified implementation
        return timedelta(minutes=30)  # Default estimate
        
    def _get_sequence_severity_distribution(self, df: pd.DataFrame, sequence: Tuple[str, ...]) -> Dict[str, int]:
        """Get severity distribution for sequence patterns."""
        sequence_alerts = df[df['metric_name'].isin(sequence)]
        return self._get_severity_distribution(sequence_alerts)
        
    def _calculate_sequence_intervals(self, df: pd.DataFrame, sequence: Tuple[str, ...]) -> float:
        """Calculate average intervals in sequence patterns."""
        return 300.0  # Default 5 minutes
        
    def _get_sequence_examples(self, df: pd.DataFrame, sequence: Tuple[str, ...], limit: int = 3) -> List[str]:
        """Get examples of sequence patterns."""
        return [f"Sequence: {' → '.join(sequence)}"]
        
    def _get_correlation_severity_distribution(self, df: pd.DataFrame, metric1: str, metric2: str) -> Dict[str, int]:
        """Get severity distribution for correlated metrics."""
        correlated_alerts = df[df['metric_name'].isin([metric1, metric2])]
        return self._get_severity_distribution(correlated_alerts)
        
    def _get_correlation_examples(self, df: pd.DataFrame, metric1: str, metric2: str, limit: int = 3) -> List[str]:
        """Get examples of correlated alerts."""
        examples = []
        metric1_alerts = df[df['metric_name'] == metric1].head(limit)
        metric2_alerts = df[df['metric_name'] == metric2].head(limit)
        
        for _, row in metric1_alerts.iterrows():
            examples.append(f"Correlation: {metric1} ↔ {metric2} at {row['timestamp']}")
            
        return examples
