"""
Enterprise Data Validation Utilities for Spotify AI Agent Alert Algorithms

This module provides comprehensive data validation, sanitization, and verification
capabilities specifically designed for music streaming platform data processing
and machine learning operations.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import re
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
import logging
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation checks"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    BUSINESS_RULE = "business_rule"
    ANOMALY_CHECK = "anomaly_check"
    CONSISTENCY_CHECK = "consistency_check"


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Definition of a validation rule"""
    name: str
    validation_type: ValidationType
    validator: Callable[[Any], bool]
    error_message: str
    is_critical: bool = True
    description: str = ""
    music_streaming_context: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_quality: DataQuality = DataQuality.EXCELLENT
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    corrected_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC):
    """Base class for all validators"""
    
    def __init__(self, name: str):
        self.name = name
        self.rules: List[ValidationRule] = []
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data according to rules"""
        pass
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules.append(rule)
        logger.debug(f"Added validation rule '{rule.name}' to validator '{self.name}'")


class MusicStreamingMetricsValidator(BaseValidator):
    """Validator for music streaming metrics data"""
    
    def __init__(self):
        super().__init__("MusicStreamingMetricsValidator")
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize validation rules for music streaming metrics"""
        
        # Audio quality validation rules
        self.add_rule(ValidationRule(
            name="audio_bitrate_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 32 <= float(x.get('audio_bitrate', 0)) <= 320,
            error_message="Audio bitrate must be between 32-320 kbps",
            music_streaming_context="audio_quality"
        ))
        
        self.add_rule(ValidationRule(
            name="audio_latency_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('audio_latency', 0)) <= 1000,
            error_message="Audio latency must be between 0-1000ms",
            music_streaming_context="audio_quality"
        ))
        
        self.add_rule(ValidationRule(
            name="buffering_ratio_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('buffering_ratio', 0)) <= 1,
            error_message="Buffering ratio must be between 0-1",
            music_streaming_context="audio_quality"
        ))
        
        # User engagement validation rules
        self.add_rule(ValidationRule(
            name="skip_rate_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('skip_rate', 0)) <= 1,
            error_message="Skip rate must be between 0-1",
            music_streaming_context="user_engagement"
        ))
        
        self.add_rule(ValidationRule(
            name="session_duration_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('session_duration', 0)) <= 86400,
            error_message="Session duration must be between 0-86400 seconds (24 hours)",
            music_streaming_context="user_engagement"
        ))
        
        self.add_rule(ValidationRule(
            name="user_retention_rate_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('user_retention_rate', 0)) <= 1,
            error_message="User retention rate must be between 0-1",
            music_streaming_context="user_engagement"
        ))
        
        # Business metrics validation rules
        self.add_rule(ValidationRule(
            name="revenue_per_user_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('revenue_per_user', 0)) <= 1000,
            error_message="Revenue per user must be between $0-$1000",
            music_streaming_context="business_metrics"
        ))
        
        self.add_rule(ValidationRule(
            name="cdn_response_time_range",
            validation_type=ValidationType.RANGE_CHECK,
            validator=lambda x: 0 <= float(x.get('cdn_response_time', 0)) <= 5000,
            error_message="CDN response time must be between 0-5000ms",
            music_streaming_context="infrastructure"
        ))
        
        # Business rule validations
        self.add_rule(ValidationRule(
            name="premium_quality_consistency",
            validation_type=ValidationType.BUSINESS_RULE,
            validator=self._validate_premium_quality_consistency,
            error_message="Premium users should have high audio quality (>= 256 kbps)",
            is_critical=False,
            music_streaming_context="business_logic"
        ))
        
        self.add_rule(ValidationRule(
            name="engagement_quality_correlation",
            validation_type=ValidationType.BUSINESS_RULE,
            validator=self._validate_engagement_quality_correlation,
            error_message="Low audio quality should correlate with higher skip rates",
            is_critical=False,
            music_streaming_context="business_logic"
        ))
    
    def _validate_premium_quality_consistency(self, data: Dict[str, Any]) -> bool:
        """Validate that premium users get appropriate audio quality"""
        user_segment = data.get('user_segment', '').lower()
        audio_bitrate = float(data.get('audio_bitrate', 0))
        
        if user_segment in ['premium', 'family', 'student']:
            return audio_bitrate >= 256
        return True  # Non-premium users can have any quality
    
    def _validate_engagement_quality_correlation(self, data: Dict[str, Any]) -> bool:
        """Validate correlation between audio quality and user engagement"""
        audio_bitrate = float(data.get('audio_bitrate', 320))
        skip_rate = float(data.get('skip_rate', 0))
        buffering_ratio = float(data.get('buffering_ratio', 0))
        
        # Low quality should correlate with higher skip rates
        if audio_bitrate < 128 or buffering_ratio > 0.1:
            return skip_rate > 0.1  # Expect higher skip rate with quality issues
        return True
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate music streaming metrics data"""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check for required fields
            required_fields = ['audio_bitrate', 'audio_latency', 'skip_rate', 'session_duration']
            missing_fields = [field for field in required_fields if field not in data or data[field] is None]
            
            if missing_fields:
                result.errors.append(f"Missing required fields: {', '.join(missing_fields)}")
                result.is_valid = False
                result.data_quality = DataQuality.CRITICAL
                return result
            
            # Apply validation rules
            critical_errors = []
            warnings = []
            
            for rule in self.rules:
                try:
                    is_valid = rule.validator(data)
                    if not is_valid:
                        if rule.is_critical:
                            critical_errors.append(rule.error_message)
                        else:
                            warnings.append(rule.error_message)
                except Exception as e:
                    error_msg = f"Validation rule '{rule.name}' failed: {str(e)}"
                    if rule.is_critical:
                        critical_errors.append(error_msg)
                    else:
                        warnings.append(error_msg)
            
            result.errors.extend(critical_errors)
            result.warnings.extend(warnings)
            result.is_valid = len(critical_errors) == 0
            
            # Determine data quality
            if critical_errors:
                result.data_quality = DataQuality.CRITICAL
            elif len(warnings) > 3:
                result.data_quality = DataQuality.POOR
            elif len(warnings) > 1:
                result.data_quality = DataQuality.ACCEPTABLE
            elif len(warnings) == 1:
                result.data_quality = DataQuality.GOOD
            else:
                result.data_quality = DataQuality.EXCELLENT
            
            # Add validation summary
            result.validation_summary = {
                'total_rules_checked': len(self.rules),
                'critical_errors': len(critical_errors),
                'warnings': len(warnings),
                'data_completeness': self._calculate_completeness(data),
                'music_streaming_context_checks': self._get_context_summary(),
            }
            
            # Attempt to correct data if needed
            if not result.is_valid:
                result.corrected_data = self._attempt_data_correction(data, result.errors)
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            result.is_valid = False
            result.errors.append(f"Validation process failed: {str(e)}")
            result.data_quality = DataQuality.CRITICAL
        
        return result
    
    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        expected_fields = [
            'audio_bitrate', 'audio_latency', 'buffering_ratio', 'skip_rate',
            'session_duration', 'user_retention_rate', 'cdn_response_time', 'revenue_per_user'
        ]
        
        present_fields = sum(1 for field in expected_fields if field in data and data[field] is not None)
        return present_fields / len(expected_fields)
    
    def _get_context_summary(self) -> Dict[str, int]:
        """Get summary of music streaming context checks"""
        context_count = {}
        for rule in self.rules:
            if rule.music_streaming_context:
                context_count[rule.music_streaming_context] = context_count.get(rule.music_streaming_context, 0) + 1
        return context_count
    
    def _attempt_data_correction(self, data: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Attempt to correct data based on common issues"""
        corrected = data.copy()
        
        # Correct out-of-range values
        if 'audio_bitrate' in corrected:
            bitrate = float(corrected['audio_bitrate'])
            if bitrate < 32:
                corrected['audio_bitrate'] = 32
            elif bitrate > 320:
                corrected['audio_bitrate'] = 320
        
        if 'skip_rate' in corrected:
            skip_rate = float(corrected['skip_rate'])
            if skip_rate < 0:
                corrected['skip_rate'] = 0
            elif skip_rate > 1:
                corrected['skip_rate'] = 1
        
        if 'buffering_ratio' in corrected:
            buffering_ratio = float(corrected['buffering_ratio'])
            if buffering_ratio < 0:
                corrected['buffering_ratio'] = 0
            elif buffering_ratio > 1:
                corrected['buffering_ratio'] = 1
        
        return corrected


class AlertDataValidator(BaseValidator):
    """Validator for alert data"""
    
    def __init__(self):
        super().__init__("AlertDataValidator")
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize validation rules for alert data"""
        
        self.add_rule(ValidationRule(
            name="alert_severity_valid",
            validation_type=ValidationType.FORMAT_CHECK,
            validator=lambda x: x.get('severity', '').lower() in ['low', 'medium', 'high', 'critical', 'emergency'],
            error_message="Alert severity must be one of: low, medium, high, critical, emergency"
        ))
        
        self.add_rule(ValidationRule(
            name="timestamp_format",
            validation_type=ValidationType.FORMAT_CHECK,
            validator=self._validate_timestamp,
            error_message="Timestamp must be a valid ISO format datetime"
        ))
        
        self.add_rule(ValidationRule(
            name="alert_source_required",
            validation_type=ValidationType.REQUIRED,
            validator=lambda x: 'source' in x and x['source'].strip() != '',
            error_message="Alert source is required and cannot be empty"
        ))
        
        self.add_rule(ValidationRule(
            name="metric_values_numeric",
            validation_type=ValidationType.TYPE_CHECK,
            validator=self._validate_metric_values,
            error_message="All metric values must be numeric"
        ))
        
        self.add_rule(ValidationRule(
            name="alert_age_reasonable",
            validation_type=ValidationType.BUSINESS_RULE,
            validator=self._validate_alert_age,
            error_message="Alert timestamp should not be more than 24 hours old",
            is_critical=False
        ))
    
    def _validate_timestamp(self, data: Dict[str, Any]) -> bool:
        """Validate timestamp format"""
        timestamp = data.get('timestamp')
        if not timestamp:
            return False
        
        try:
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                return True
            else:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_metric_values(self, data: Dict[str, Any]) -> bool:
        """Validate that metric values are numeric"""
        metrics = data.get('metrics', {})
        if not isinstance(metrics, dict):
            return False
        
        for key, value in metrics.items():
            try:
                float(value)
            except (ValueError, TypeError):
                if value is not None:  # Allow None values
                    return False
        return True
    
    def _validate_alert_age(self, data: Dict[str, Any]) -> bool:
        """Validate that alert is not too old"""
        timestamp = data.get('timestamp')
        if not timestamp:
            return False
        
        try:
            if isinstance(timestamp, str):
                alert_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                alert_time = timestamp
            
            age = datetime.now() - alert_time.replace(tzinfo=None)
            return age <= timedelta(hours=24)
        except:
            return False
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate alert data"""
        result = ValidationResult(is_valid=True)
        
        try:
            # Apply all validation rules
            for rule in self.rules:
                try:
                    is_valid = rule.validator(data)
                    if not is_valid:
                        if rule.is_critical:
                            result.errors.append(rule.error_message)
                        else:
                            result.warnings.append(rule.error_message)
                except Exception as e:
                    error_msg = f"Validation rule '{rule.name}' failed: {str(e)}"
                    if rule.is_critical:
                        result.errors.append(error_msg)
                    else:
                        result.warnings.append(error_msg)
            
            result.is_valid = len(result.errors) == 0
            
            # Determine data quality
            if result.errors:
                result.data_quality = DataQuality.CRITICAL if len(result.errors) > 2 else DataQuality.POOR
            elif result.warnings:
                result.data_quality = DataQuality.ACCEPTABLE if len(result.warnings) > 1 else DataQuality.GOOD
            else:
                result.data_quality = DataQuality.EXCELLENT
            
            result.validation_summary = {
                'total_rules_checked': len(self.rules),
                'errors': len(result.errors),
                'warnings': len(result.warnings),
            }
            
        except Exception as e:
            logger.error(f"Error during alert validation: {e}")
            result.is_valid = False
            result.errors.append(f"Alert validation failed: {str(e)}")
            result.data_quality = DataQuality.CRITICAL
        
        return result


class MLModelInputValidator(BaseValidator):
    """Validator for machine learning model inputs"""
    
    def __init__(self, expected_features: Optional[List[str]] = None):
        super().__init__("MLModelInputValidator")
        self.expected_features = expected_features or []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize validation rules for ML model inputs"""
        
        self.add_rule(ValidationRule(
            name="feature_completeness",
            validation_type=ValidationType.REQUIRED,
            validator=self._validate_feature_completeness,
            error_message="All expected features must be present"
        ))
        
        self.add_rule(ValidationRule(
            name="numeric_features_valid",
            validation_type=ValidationType.TYPE_CHECK,
            validator=self._validate_numeric_features,
            error_message="Numeric features must contain valid numbers"
        ))
        
        self.add_rule(ValidationRule(
            name="no_infinite_values",
            validation_type=ValidationType.ANOMALY_CHECK,
            validator=self._validate_no_infinite_values,
            error_message="Features cannot contain infinite values"
        ))
        
        self.add_rule(ValidationRule(
            name="feature_distributions_normal",
            validation_type=ValidationType.ANOMALY_CHECK,
            validator=self._validate_feature_distributions,
            error_message="Feature values appear to be outside normal distributions",
            is_critical=False
        ))
    
    def _validate_feature_completeness(self, data: Any) -> bool:
        """Validate that all expected features are present"""
        if not self.expected_features:
            return True
        
        if isinstance(data, dict):
            return all(feature in data for feature in self.expected_features)
        elif isinstance(data, (list, np.ndarray)):
            return len(data) >= len(self.expected_features)
        elif isinstance(data, pd.DataFrame):
            return all(feature in data.columns for feature in self.expected_features)
        
        return False
    
    def _validate_numeric_features(self, data: Any) -> bool:
        """Validate numeric features"""
        try:
            if isinstance(data, dict):
                for value in data.values():
                    if value is not None:
                        float(value)
            elif isinstance(data, (list, np.ndarray)):
                for value in data:
                    if value is not None:
                        float(value)
            elif isinstance(data, pd.DataFrame):
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                return not data[numeric_columns].isnull().all().any()
            
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_no_infinite_values(self, data: Any) -> bool:
        """Validate no infinite values"""
        try:
            if isinstance(data, dict):
                values = [float(v) for v in data.values() if v is not None]
            elif isinstance(data, (list, np.ndarray)):
                values = [float(v) for v in data if v is not None]
            elif isinstance(data, pd.DataFrame):
                values = data.select_dtypes(include=[np.number]).values.flatten()
            else:
                return True
            
            return not any(np.isinf(v) for v in values if not np.isnan(v))
        except:
            return False
    
    def _validate_feature_distributions(self, data: Any) -> bool:
        """Validate feature distributions are reasonable"""
        try:
            if isinstance(data, dict):
                values = [float(v) for v in data.values() if v is not None]
            elif isinstance(data, (list, np.ndarray)):
                values = [float(v) for v in data if v is not None]
            elif isinstance(data, pd.DataFrame):
                # For DataFrame, check each numeric column
                for column in data.select_dtypes(include=[np.number]).columns:
                    col_values = data[column].dropna().values
                    if len(col_values) > 0:
                        # Check for extreme outliers (> 5 standard deviations)
                        mean_val = np.mean(col_values)
                        std_val = np.std(col_values)
                        if std_val > 0:
                            outliers = np.abs(col_values - mean_val) > 5 * std_val
                            if np.sum(outliers) / len(col_values) > 0.1:  # More than 10% outliers
                                return False
                return True
            else:
                return True
            
            if len(values) == 0:
                return True
            
            # Check for extreme outliers
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > 0:
                outliers = [abs(v - mean_val) > 5 * std_val for v in values]
                return sum(outliers) / len(values) <= 0.1  # Less than 10% outliers
            
            return True
        except:
            return True  # Don't fail on distribution check errors
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate ML model input data"""
        result = ValidationResult(is_valid=True)
        
        try:
            # Apply validation rules
            for rule in self.rules:
                try:
                    is_valid = rule.validator(data)
                    if not is_valid:
                        if rule.is_critical:
                            result.errors.append(rule.error_message)
                        else:
                            result.warnings.append(rule.error_message)
                except Exception as e:
                    error_msg = f"ML validation rule '{rule.name}' failed: {str(e)}"
                    if rule.is_critical:
                        result.errors.append(error_msg)
                    else:
                        result.warnings.append(error_msg)
            
            result.is_valid = len(result.errors) == 0
            
            # Data quality assessment
            if result.errors:
                result.data_quality = DataQuality.CRITICAL
            elif len(result.warnings) > 1:
                result.data_quality = DataQuality.POOR
            elif len(result.warnings) == 1:
                result.data_quality = DataQuality.ACCEPTABLE
            else:
                result.data_quality = DataQuality.EXCELLENT
            
            # Add metadata about the input
            if isinstance(data, pd.DataFrame):
                result.metadata = {
                    'shape': data.shape,
                    'dtypes': data.dtypes.to_dict(),
                    'missing_values': data.isnull().sum().to_dict()
                }
            elif isinstance(data, dict):
                result.metadata = {
                    'feature_count': len(data),
                    'features': list(data.keys())
                }
            elif isinstance(data, (list, np.ndarray)):
                result.metadata = {
                    'length': len(data),
                    'type': str(type(data))
                }
            
        except Exception as e:
            logger.error(f"Error during ML input validation: {e}")
            result.is_valid = False
            result.errors.append(f"ML input validation failed: {str(e)}")
            result.data_quality = DataQuality.CRITICAL
        
        return result


class ValidationManager:
    """Central manager for all validation operations"""
    
    def __init__(self):
        self.validators: Dict[str, BaseValidator] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators"""
        self.register_validator("music_streaming_metrics", MusicStreamingMetricsValidator())
        self.register_validator("alert_data", AlertDataValidator())
        self.register_validator("ml_input", MLModelInputValidator())
    
    def register_validator(self, name: str, validator: BaseValidator):
        """Register a validator"""
        self.validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def validate(self, validator_name: str, data: Any) -> ValidationResult:
        """Validate data using specified validator"""
        if validator_name not in self.validators:
            result = ValidationResult(is_valid=False)
            result.errors.append(f"Validator '{validator_name}' not found")
            result.data_quality = DataQuality.CRITICAL
            return result
        
        return self.validators[validator_name].validate(data)
    
    def validate_multiple(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate data using multiple validators"""
        results = {}
        for validator_name, validator_data in data.items():
            if validator_name in self.validators:
                results[validator_name] = self.validate(validator_name, validator_data)
            else:
                result = ValidationResult(is_valid=False)
                result.errors.append(f"Validator '{validator_name}' not found")
                results[validator_name] = result
        
        return results
    
    def get_overall_quality(self, results: Dict[str, ValidationResult]) -> DataQuality:
        """Get overall data quality from multiple validation results"""
        if not results:
            return DataQuality.CRITICAL
        
        quality_scores = {
            DataQuality.EXCELLENT: 5,
            DataQuality.GOOD: 4,
            DataQuality.ACCEPTABLE: 3,
            DataQuality.POOR: 2,
            DataQuality.CRITICAL: 1
        }
        
        # Calculate weighted average (equal weights for now)
        total_score = sum(quality_scores[result.data_quality] for result in results.values())
        avg_score = total_score / len(results)
        
        # Map back to quality level
        if avg_score >= 4.5:
            return DataQuality.EXCELLENT
        elif avg_score >= 3.5:
            return DataQuality.GOOD
        elif avg_score >= 2.5:
            return DataQuality.ACCEPTABLE
        elif avg_score >= 1.5:
            return DataQuality.POOR
        else:
            return DataQuality.CRITICAL


# Global validation manager instance
_validation_manager: Optional[ValidationManager] = None


def get_validation_manager() -> ValidationManager:
    """Get or create global validation manager instance"""
    global _validation_manager
    
    if _validation_manager is None:
        _validation_manager = ValidationManager()
    
    return _validation_manager


def validate_music_streaming_data(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate music streaming data"""
    manager = get_validation_manager()
    return manager.validate("music_streaming_metrics", data)


def validate_alert_data(data: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate alert data"""
    manager = get_validation_manager()
    return manager.validate("alert_data", data)


def validate_ml_input(data: Any, expected_features: Optional[List[str]] = None) -> ValidationResult:
    """Convenience function to validate ML model input"""
    manager = get_validation_manager()
    
    # Create a temporary validator with expected features if provided
    if expected_features:
        validator = MLModelInputValidator(expected_features)
        manager.register_validator("temp_ml_input", validator)
        result = manager.validate("temp_ml_input", data)
        # Clean up temporary validator
        del manager.validators["temp_ml_input"]
        return result
    else:
        return manager.validate("ml_input", data)
