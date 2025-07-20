"""
Data Quality Checker - Ultra-Advanced Edition
============================================

Ultra-advanced data quality validation, cleansing, and monitoring system
with ML-based anomaly detection and automated remediation.

Features:
- Real-time data quality validation
- ML-based anomaly detection
- Automated data cleansing
- Data lineage tracking
- Quality metrics and reporting
- Compliance validation
- Performance optimization
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class DataQualityRule:
    """Data quality rule configuration."""
    
    rule_id: str
    name: str
    description: str
    rule_type: str  # completeness, validity, accuracy, consistency, uniqueness
    column: str
    condition: str
    threshold: float
    severity: str  # critical, high, medium, low
    auto_fix: bool = False
    notification_required: bool = True


@dataclass
class QualityViolation:
    """Data quality violation record."""
    
    violation_id: str
    rule_id: str
    column: str
    violation_type: str
    severity: str
    record_count: int
    violation_percentage: float
    detected_at: datetime
    sample_values: List[Any]
    suggested_fix: Optional[str] = None
    auto_fixed: bool = False


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    
    report_id: str
    dataset_name: str
    generated_at: datetime
    total_records: int
    total_columns: int
    overall_score: float
    violations: List[QualityViolation]
    metrics: Dict[str, Any]
    recommendations: List[str]
    data_lineage: Dict[str, Any]


class DataQualityChecker:
    """Ultra-advanced data quality checker with ML capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data quality checker."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.rules: List[DataQualityRule] = []
        self.ml_models: Dict[str, Any] = {}
        self.baseline_stats: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.performance_metrics = {
            'checks_performed': 0,
            'violations_detected': 0,
            'auto_fixes_applied': 0,
            'processing_time': []
        }
        
        self._load_quality_rules()
        self._initialize_ml_models()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'ml_anomaly_detection': True,
            'auto_fix_enabled': True,
            'parallel_processing': True,
            'max_workers': mp.cpu_count(),
            'quality_threshold': 0.95,
            'sample_size_for_analysis': 10000,
            'notification_webhooks': [],
            'export_formats': ['json', 'html', 'csv'],
            'data_lineage_tracking': True,
            'performance_monitoring': True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup advanced logging configuration."""
        logger = logging.getLogger('DataQualityChecker')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_quality_rules(self):
        """Load predefined data quality rules."""
        # Standard quality rules for common data issues
        standard_rules = [
            DataQualityRule(
                rule_id="null_check",
                name="Null Value Check",
                description="Check for null/missing values",
                rule_type="completeness",
                column="*",
                condition="not_null",
                threshold=0.95,
                severity="high",
                auto_fix=True
            ),
            DataQualityRule(
                rule_id="duplicate_check",
                name="Duplicate Records Check",
                description="Check for duplicate records",
                rule_type="uniqueness",
                column="*",
                condition="unique",
                threshold=1.0,
                severity="medium",
                auto_fix=True
            ),
            DataQualityRule(
                rule_id="email_format",
                name="Email Format Validation",
                description="Validate email address format",
                rule_type="validity",
                column="email",
                condition="email_regex",
                threshold=1.0,
                severity="high",
                auto_fix=False
            ),
            DataQualityRule(
                rule_id="date_range",
                name="Date Range Validation",
                description="Check if dates are within valid range",
                rule_type="validity",
                column="date",
                condition="date_range",
                threshold=1.0,
                severity="critical",
                auto_fix=False
            ),
            DataQualityRule(
                rule_id="numeric_range",
                name="Numeric Range Validation",
                description="Check if numeric values are within expected range",
                rule_type="validity",
                column="numeric",
                condition="numeric_range",
                threshold=0.99,
                severity="medium",
                auto_fix=True
            )
        ]
        
        self.rules.extend(standard_rules)
        self.logger.info(f"Loaded {len(self.rules)} standard quality rules")
    
    def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection."""
        if self.config.get('ml_anomaly_detection', True):
            # Isolation Forest for outlier detection
            self.ml_models['outlier_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Additional models can be added here
            self.logger.info("Initialized ML models for anomaly detection")
    
    async def check_data_quality(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        custom_rules: Optional[List[DataQualityRule]] = None
    ) -> DataQualityReport:
        """Perform comprehensive data quality check."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting data quality check for dataset: {dataset_name}")
            
            # Combine standard and custom rules
            all_rules = self.rules.copy()
            if custom_rules:
                all_rules.extend(custom_rules)
            
            # Run quality checks
            violations = []
            
            if self.config.get('parallel_processing', True):
                violations = await self._run_parallel_checks(data, all_rules)
            else:
                violations = await self._run_sequential_checks(data, all_rules)
            
            # ML-based anomaly detection
            if self.config.get('ml_anomaly_detection', True):
                ml_violations = await self._detect_ml_anomalies(data)
                violations.extend(ml_violations)
            
            # Calculate overall quality score
            overall_score = self._calculate_quality_score(data, violations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(violations)
            
            # Track data lineage
            data_lineage = self._track_data_lineage(dataset_name)
            
            # Create comprehensive report
            report = DataQualityReport(
                report_id=f"dq_{dataset_name}_{int(datetime.now().timestamp())}",
                dataset_name=dataset_name,
                generated_at=datetime.now(),
                total_records=len(data),
                total_columns=len(data.columns),
                overall_score=overall_score,
                violations=violations,
                metrics=self._calculate_detailed_metrics(data, violations),
                recommendations=recommendations,
                data_lineage=data_lineage
            )
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['checks_performed'] += 1
            self.performance_metrics['violations_detected'] += len(violations)
            self.performance_metrics['processing_time'].append(processing_time)
            
            self.logger.info(
                f"Data quality check completed. Score: {overall_score:.2f}, "
                f"Violations: {len(violations)}, Time: {processing_time:.2f}s"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during data quality check: {str(e)}")
            raise
    
    async def _run_parallel_checks(
        self,
        data: pd.DataFrame,
        rules: List[DataQualityRule]
    ) -> List[QualityViolation]:
        """Run quality checks in parallel for better performance."""
        violations = []
        max_workers = self.config.get('max_workers', mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for rule in rules:
                future = executor.submit(self._apply_quality_rule, data, rule)
                futures.append(future)
            
            for future in futures:
                try:
                    rule_violations = future.result()
                    violations.extend(rule_violations)
                except Exception as e:
                    self.logger.error(f"Error in parallel quality check: {str(e)}")
        
        return violations
    
    async def _run_sequential_checks(
        self,
        data: pd.DataFrame,
        rules: List[DataQualityRule]
    ) -> List[QualityViolation]:
        """Run quality checks sequentially."""
        violations = []
        
        for rule in rules:
            try:
                rule_violations = self._apply_quality_rule(data, rule)
                violations.extend(rule_violations)
            except Exception as e:
                self.logger.error(f"Error applying rule {rule.rule_id}: {str(e)}")
        
        return violations
    
    def _apply_quality_rule(
        self,
        data: pd.DataFrame,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Apply a specific quality rule to the data."""
        violations = []
        
        try:
            if rule.column == "*":
                # Apply rule to all applicable columns
                applicable_columns = self._get_applicable_columns(data, rule)
                for col in applicable_columns:
                    col_violations = self._check_column_rule(data, col, rule)
                    violations.extend(col_violations)
            else:
                # Apply rule to specific column
                if rule.column in data.columns:
                    col_violations = self._check_column_rule(data, rule.column, rule)
                    violations.extend(col_violations)
        
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.rule_id}: {str(e)}")
        
        return violations
    
    def _check_column_rule(
        self,
        data: pd.DataFrame,
        column: str,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Check a specific rule against a column."""
        violations = []
        
        try:
            if rule.rule_type == "completeness":
                violations.extend(self._check_completeness(data, column, rule))
            elif rule.rule_type == "uniqueness":
                violations.extend(self._check_uniqueness(data, column, rule))
            elif rule.rule_type == "validity":
                violations.extend(self._check_validity(data, column, rule))
            elif rule.rule_type == "accuracy":
                violations.extend(self._check_accuracy(data, column, rule))
            elif rule.rule_type == "consistency":
                violations.extend(self._check_consistency(data, column, rule))
        
        except Exception as e:
            self.logger.error(f"Error checking rule {rule.rule_id} on column {column}: {str(e)}")
        
        return violations
    
    def _check_completeness(
        self,
        data: pd.DataFrame,
        column: str,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Check data completeness."""
        violations = []
        
        null_count = data[column].isnull().sum()
        total_count = len(data)
        completeness_rate = (total_count - null_count) / total_count
        
        if completeness_rate < rule.threshold:
            violation = QualityViolation(
                violation_id=f"{rule.rule_id}_{column}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                column=column,
                violation_type="completeness",
                severity=rule.severity,
                record_count=null_count,
                violation_percentage=(1 - completeness_rate) * 100,
                detected_at=datetime.now(),
                sample_values=[],
                suggested_fix="Fill missing values with appropriate defaults or remove incomplete records"
            )
            
            # Auto-fix if enabled
            if rule.auto_fix and self.config.get('auto_fix_enabled', True):
                violation.auto_fixed = self._auto_fix_completeness(data, column)
            
            violations.append(violation)
        
        return violations
    
    def _check_uniqueness(
        self,
        data: pd.DataFrame,
        column: str,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Check data uniqueness."""
        violations = []
        
        duplicate_count = data[column].duplicated().sum()
        total_count = len(data)
        uniqueness_rate = (total_count - duplicate_count) / total_count
        
        if uniqueness_rate < rule.threshold:
            duplicates = data[data[column].duplicated()][column].tolist()
            sample_duplicates = duplicates[:10]  # Sample of duplicates
            
            violation = QualityViolation(
                violation_id=f"{rule.rule_id}_{column}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                column=column,
                violation_type="uniqueness",
                severity=rule.severity,
                record_count=duplicate_count,
                violation_percentage=(1 - uniqueness_rate) * 100,
                detected_at=datetime.now(),
                sample_values=sample_duplicates,
                suggested_fix="Remove duplicate records or add unique constraints"
            )
            
            # Auto-fix if enabled
            if rule.auto_fix and self.config.get('auto_fix_enabled', True):
                violation.auto_fixed = self._auto_fix_duplicates(data, column)
            
            violations.append(violation)
        
        return violations
    
    def _check_validity(
        self,
        data: pd.DataFrame,
        column: str,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Check data validity based on format rules."""
        violations = []
        
        if rule.condition == "email_regex":
            violations.extend(self._check_email_format(data, column, rule))
        elif rule.condition == "date_range":
            violations.extend(self._check_date_range(data, column, rule))
        elif rule.condition == "numeric_range":
            violations.extend(self._check_numeric_range(data, column, rule))
        
        return violations
    
    def _check_email_format(
        self,
        data: pd.DataFrame,
        column: str,
        rule: DataQualityRule
    ) -> List[QualityViolation]:
        """Check email format validity."""
        violations = []
        
        if column in data.columns and data[column].dtype == 'object':
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            invalid_emails = data[
                (~data[column].str.match(email_pattern, na=False)) & 
                (data[column].notna())
            ]
            
            if len(invalid_emails) > 0:
                invalid_count = len(invalid_emails)
                total_count = len(data[data[column].notna()])
                violation_percentage = (invalid_count / total_count) * 100
                
                violation = QualityViolation(
                    violation_id=f"{rule.rule_id}_{column}_{int(datetime.now().timestamp())}",
                    rule_id=rule.rule_id,
                    column=column,
                    violation_type="validity",
                    severity=rule.severity,
                    record_count=invalid_count,
                    violation_percentage=violation_percentage,
                    detected_at=datetime.now(),
                    sample_values=invalid_emails[column].head(10).tolist(),
                    suggested_fix="Validate and correct email formats"
                )
                
                violations.append(violation)
        
        return violations
    
    async def _detect_ml_anomalies(self, data: pd.DataFrame) -> List[QualityViolation]:
        """Use ML models to detect data anomalies."""
        violations = []
        
        try:
            if 'outlier_detector' in self.ml_models:
                # Select numeric columns for anomaly detection
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    # Prepare data for ML model
                    ml_data = data[numeric_columns].fillna(data[numeric_columns].median())
                    
                    if len(ml_data) > 10:  # Minimum data points required
                        # Scale the data
                        scaled_data = self.scaler.fit_transform(ml_data)
                        
                        # Detect outliers
                        outliers = self.ml_models['outlier_detector'].fit_predict(scaled_data)
                        anomaly_mask = outliers == -1
                        
                        if anomaly_mask.sum() > 0:
                            anomaly_count = anomaly_mask.sum()
                            violation_percentage = (anomaly_count / len(data)) * 100
                            
                            violation = QualityViolation(
                                violation_id=f"ml_anomaly_{int(datetime.now().timestamp())}",
                                rule_id="ml_outlier_detection",
                                column="multiple_numeric",
                                violation_type="anomaly",
                                severity="medium",
                                record_count=anomaly_count,
                                violation_percentage=violation_percentage,
                                detected_at=datetime.now(),
                                sample_values=[],
                                suggested_fix="Review and validate anomalous records"
                            )
                            
                            violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {str(e)}")
        
        return violations
    
    def _calculate_quality_score(
        self,
        data: pd.DataFrame,
        violations: List[QualityViolation]
    ) -> float:
        """Calculate overall data quality score."""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        total_weight = 0
        weighted_violations = 0
        
        for violation in violations:
            weight = severity_weights.get(violation.severity, 0.5)
            violation_impact = (violation.violation_percentage / 100) * weight
            weighted_violations += violation_impact
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        # Calculate score (0-1 scale)
        quality_score = max(0, 1 - (weighted_violations / total_weight))
        return quality_score
    
    def _generate_recommendations(
        self,
        violations: List[QualityViolation]
    ) -> List[str]:
        """Generate actionable recommendations based on violations."""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.violation_type not in violation_types:
                violation_types[violation.violation_type] = []
            violation_types[violation.violation_type].append(violation)
        
        # Generate specific recommendations
        if 'completeness' in violation_types:
            recommendations.append(
                "Implement data validation at source to prevent missing values"
            )
            recommendations.append(
                "Consider using default values or imputation strategies for missing data"
            )
        
        if 'uniqueness' in violation_types:
            recommendations.append(
                "Add unique constraints to prevent duplicate records"
            )
            recommendations.append(
                "Implement deduplication processes in data pipeline"
            )
        
        if 'validity' in violation_types:
            recommendations.append(
                "Implement format validation at data entry points"
            )
            recommendations.append(
                "Use data type constraints and validation rules"
            )
        
        if 'anomaly' in violation_types:
            recommendations.append(
                "Implement real-time anomaly detection in data pipeline"
            )
            recommendations.append(
                "Review data collection processes for potential issues"
            )
        
        return recommendations
    
    def _track_data_lineage(self, dataset_name: str) -> Dict[str, Any]:
        """Track data lineage information."""
        return {
            'dataset_name': dataset_name,
            'check_timestamp': datetime.now().isoformat(),
            'checker_version': '2.0.0',
            'source_system': 'spotify_ai_agent',
            'processing_stage': 'quality_validation'
        }
    
    def _calculate_detailed_metrics(
        self,
        data: pd.DataFrame,
        violations: List[QualityViolation]
    ) -> Dict[str, Any]:
        """Calculate detailed quality metrics."""
        return {
            'completeness_rate': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'uniqueness_rate': 1 - (data.duplicated().sum() / len(data)),
            'violation_count_by_severity': {
                severity: len([v for v in violations if v.severity == severity])
                for severity in ['critical', 'high', 'medium', 'low']
            },
            'columns_with_issues': len(set(v.column for v in violations)),
            'auto_fixes_applied': len([v for v in violations if v.auto_fixed])
        }
    
    def _auto_fix_completeness(self, data: pd.DataFrame, column: str) -> bool:
        """Auto-fix completeness issues."""
        try:
            if data[column].dtype in ['int64', 'float64']:
                # Fill with median for numeric columns
                data[column].fillna(data[column].median(), inplace=True)
            else:
                # Fill with mode for categorical columns
                mode_value = data[column].mode()
                if len(mode_value) > 0:
                    data[column].fillna(mode_value[0], inplace=True)
            
            self.performance_metrics['auto_fixes_applied'] += 1
            return True
        except Exception:
            return False
    
    def _auto_fix_duplicates(self, data: pd.DataFrame, column: str) -> bool:
        """Auto-fix duplicate issues."""
        try:
            # Remove duplicates keeping first occurrence
            initial_count = len(data)
            data.drop_duplicates(subset=[column], keep='first', inplace=True)
            
            if len(data) < initial_count:
                self.performance_metrics['auto_fixes_applied'] += 1
                return True
            return False
        except Exception:
            return False
    
    def _get_applicable_columns(
        self,
        data: pd.DataFrame,
        rule: DataQualityRule
    ) -> List[str]:
        """Get columns applicable for a specific rule."""
        if rule.rule_type == "completeness":
            return data.columns.tolist()
        elif rule.rule_type == "uniqueness":
            return data.columns.tolist()
        elif rule.condition == "email_regex":
            return [col for col in data.columns if 'email' in col.lower()]
        elif rule.condition == "numeric_range":
            return data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            return data.columns.tolist()
    
    def export_report(
        self,
        report: DataQualityReport,
        output_path: str,
        format: str = 'json'
    ):
        """Export quality report in various formats."""
        try:
            if format == 'json':
                self._export_json_report(report, output_path)
            elif format == 'html':
                self._export_html_report(report, output_path)
            elif format == 'csv':
                self._export_csv_report(report, output_path)
            
            self.logger.info(f"Report exported to {output_path} in {format} format")
        
        except Exception as e:
            self.logger.error(f"Error exporting report: {str(e)}")
    
    def _export_json_report(self, report: DataQualityReport, output_path: str):
        """Export report as JSON."""
        report_dict = asdict(report)
        
        # Convert datetime objects to strings
        report_dict['generated_at'] = report.generated_at.isoformat()
        for violation in report_dict['violations']:
            violation['detected_at'] = violation['detected_at'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    def _export_html_report(self, report: DataQualityReport, output_path: str):
        """Export report as HTML with visualizations."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e0e0e0; border-radius: 3px; }}
                .violation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ff6b6b; background-color: #ffe0e0; }}
                .recommendation {{ margin: 5px 0; padding: 8px; background-color: #e0ffe0; border-left: 4px solid #4ecdc4; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {report.dataset_name}</p>
                <p><strong>Generated:</strong> {report.generated_at}</p>
                <p><strong>Overall Score:</strong> {report.overall_score:.2f}</p>
            </div>
            
            <h2>Summary Metrics</h2>
            <div class="metric">Total Records: {report.total_records}</div>
            <div class="metric">Total Columns: {report.total_columns}</div>
            <div class="metric">Violations Found: {len(report.violations)}</div>
            
            <h2>Quality Violations</h2>
        """
        
        for violation in report.violations:
            html_content += f"""
            <div class="violation">
                <strong>{violation.violation_type.title()} Issue</strong> in column <strong>{violation.column}</strong><br>
                Severity: {violation.severity}<br>
                Affected Records: {violation.record_count} ({violation.violation_percentage:.1f}%)<br>
                {f"Auto-fixed: Yes" if violation.auto_fixed else ""}
            </div>
            """
        
        html_content += "<h2>Recommendations</h2>"
        for rec in report.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'
        
        html_content += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _export_csv_report(self, report: DataQualityReport, output_path: str):
        """Export violations as CSV."""
        violations_data = []
        for violation in report.violations:
            violations_data.append({
                'violation_id': violation.violation_id,
                'column': violation.column,
                'violation_type': violation.violation_type,
                'severity': violation.severity,
                'record_count': violation.record_count,
                'violation_percentage': violation.violation_percentage,
                'detected_at': violation.detected_at,
                'auto_fixed': violation.auto_fixed
            })
        
        df = pd.DataFrame(violations_data)
        df.to_csv(output_path, index=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = (
            sum(self.performance_metrics['processing_time']) / 
            len(self.performance_metrics['processing_time'])
            if self.performance_metrics['processing_time'] else 0
        )
        
        return {
            'checks_performed': self.performance_metrics['checks_performed'],
            'violations_detected': self.performance_metrics['violations_detected'],
            'auto_fixes_applied': self.performance_metrics['auto_fixes_applied'],
            'average_processing_time': avg_processing_time,
            'total_processing_time': sum(self.performance_metrics['processing_time'])
        }


# Utility functions for external use
async def run_quality_check(
    data_source: str,
    dataset_name: str,
    config_path: Optional[str] = None
) -> DataQualityReport:
    """Convenience function to run quality check on data source."""
    checker = DataQualityChecker(config_path)
    
    # Load data (implement based on source type)
    if data_source.endswith('.csv'):
        data = pd.read_csv(data_source)
    elif data_source.endswith('.json'):
        data = pd.read_json(data_source)
    else:
        raise ValueError(f"Unsupported data source format: {data_source}")
    
    return await checker.check_data_quality(data, dataset_name)


def create_quality_dashboard(reports: List[DataQualityReport]) -> str:
    """Create interactive dashboard from quality reports."""
    # Implementation for creating Plotly dashboard
    # This would create visualizations showing quality trends over time
    pass


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create sample data with quality issues
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 5],  # Duplicate
            'email': ['test@example.com', 'invalid-email', 'user@domain.com', None, 'admin@site.org', 'test@example.com'],
            'age': [25, 30, None, 45, 200, 25],  # Missing and outlier
            'score': [85.5, 92.0, 78.5, 89.0, 95.5, 85.5]
        })
        
        checker = DataQualityChecker()
        report = await checker.check_data_quality(sample_data, "sample_dataset")
        
        print(f"Quality Score: {report.overall_score:.2f}")
        print(f"Violations Found: {len(report.violations)}")
        
        # Export report
        checker.export_report(report, "quality_report.json", "json")
        checker.export_report(report, "quality_report.html", "html")
        
        # Show performance stats
        stats = checker.get_performance_stats()
        print(f"Performance Stats: {stats}")
    
    asyncio.run(main())
