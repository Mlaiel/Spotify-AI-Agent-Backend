"""
Advanced Analytics Utilities for Spotify AI Agent
===============================================

Ultra-sophisticated utility functions and classes for advanced analytics,
data processing, and business intelligence operations.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, DBA & Data Engineer, Développeur Backend Senior
"""

import asyncio
import functools
import hashlib
import json
import logging
import numpy as np
import pandas as pd
import pickle
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aioredis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import boto3
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    dataset_name: str
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    quality_score: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """Performance metrics for analytics operations."""
    operation_name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    success_rate: float
    error_count: int
    timestamp: datetime = field(default_factory=datetime.now)

class DataProcessor:
    """Advanced data processing utility with enterprise features."""
    
    def __init__(self, cache_enabled: bool = True, encryption_key: Optional[bytes] = None):
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
    async def process_dataframe(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Process DataFrame with a list of operations."""
        start_time = datetime.now()
        
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Apply operations
            for operation in operations:
                processed_df = await self._apply_operation(processed_df, operation)
            
            # Log performance
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"DataFrame processing completed in {duration:.2f} seconds")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame: {e}")
            raise

    async def _apply_operation(self, df: pd.DataFrame, operation: str) -> pd.DataFrame:
        """Apply a specific operation to DataFrame."""
        operations_map = {
            'remove_duplicates': lambda x: x.drop_duplicates(),
            'fill_missing': lambda x: x.fillna(x.mean(numeric_only=True)),
            'normalize': lambda x: self._normalize_dataframe(x),
            'remove_outliers': lambda x: self._remove_outliers(x),
            'encode_categorical': lambda x: self._encode_categorical(x)
        }
        
        if operation in operations_map:
            return operations_map[operation](df)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return df

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric columns in DataFrame."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        
        df_normalized = df.copy()
        df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df_normalized

    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from DataFrame."""
        if method == 'iqr':
            return self._remove_outliers_iqr(df)
        elif method == 'zscore':
            return self._remove_outliers_zscore(df)
        else:
            return df

    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        
        for column in numeric_columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[
                (df_clean[column] >= lower_bound) & 
                (df_clean[column] <= upper_bound)
            ]
        
        return df_clean

    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        
        for column in numeric_columns:
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            df_clean = df_clean[z_scores < threshold]
        
        return df_clean

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if df[column].nunique() < 10:  # Use one-hot encoding for low cardinality
                dummies = pd.get_dummies(df[column], prefix=column)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
            else:  # Use label encoding for high cardinality
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df[column].astype(str))
        
        return df_encoded

    async def assess_data_quality(self, df: pd.DataFrame, dataset_name: str) -> DataQualityReport:
        """Comprehensive data quality assessment."""
        try:
            # Basic statistics
            total_records = len(df)
            missing_values = df.isnull().sum().to_dict()
            duplicate_records = df.duplicated().sum()
            
            # Outlier detection
            outliers = {}
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(df[
                    (df[column] < lower_bound) | (df[column] > upper_bound)
                ])
                outliers[column] = outlier_count
            
            # Data types
            data_types = df.dtypes.astype(str).to_dict()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                total_records, missing_values, duplicate_records, outliers
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                missing_values, duplicate_records, outliers, quality_score
            )
            
            return DataQualityReport(
                dataset_name=dataset_name,
                total_records=total_records,
                missing_values=missing_values,
                duplicate_records=duplicate_records,
                outliers=outliers,
                data_types=data_types,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            raise

    def _calculate_quality_score(self, total_records: int, missing_values: Dict[str, int], 
                                duplicate_records: int, outliers: Dict[str, int]) -> float:
        """Calculate overall data quality score."""
        if total_records == 0:
            return 0.0
        
        # Missing values penalty
        missing_penalty = sum(missing_values.values()) / (total_records * len(missing_values))
        
        # Duplicate penalty
        duplicate_penalty = duplicate_records / total_records
        
        # Outlier penalty
        outlier_penalty = sum(outliers.values()) / (total_records * len(outliers)) if outliers else 0
        
        # Calculate score (higher is better)
        quality_score = 1.0 - (missing_penalty + duplicate_penalty + outlier_penalty) / 3
        
        return max(0.0, min(1.0, quality_score))

    def _generate_recommendations(self, missing_values: Dict[str, int], duplicate_records: int,
                                 outliers: Dict[str, int], quality_score: float) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        if quality_score < 0.8:
            recommendations.append("Overall data quality needs improvement")
        
        # Missing values recommendations
        high_missing_columns = [col for col, count in missing_values.items() if count > 0]
        if high_missing_columns:
            recommendations.append(f"Address missing values in columns: {', '.join(high_missing_columns)}")
        
        # Duplicate recommendations
        if duplicate_records > 0:
            recommendations.append(f"Remove {duplicate_records} duplicate records")
        
        # Outlier recommendations
        high_outlier_columns = [col for col, count in outliers.items() if count > 0]
        if high_outlier_columns:
            recommendations.append(f"Investigate outliers in columns: {', '.join(high_outlier_columns)}")
        
        return recommendations

class CacheManager:
    """Advanced caching system with TTL, compression, and encryption."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None, 
                 default_ttl: int = 3600, compression_enabled: bool = True,
                 encryption_enabled: bool = True):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        self.local_cache = {}
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key) if encryption_enabled else None
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try local cache first
            if key in self.local_cache:
                data, expiry = self.local_cache[key]
                if datetime.now() < expiry:
                    return data
                else:
                    del self.local_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    return self._deserialize(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            # Store in local cache
            self.local_cache[key] = (value, expiry)
            
            # Store in Redis cache
            if self.redis_client:
                serialized_data = self._serialize(value)
                await self.redis_client.setex(key, ttl, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Remove from local cache
            if key in self.local_cache:
                del self.local_cache[key]
            
            # Remove from Redis cache
            if self.redis_client:
                await self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        # Pickle the data
        pickled_data = pickle.dumps(data)
        
        # Compress if enabled
        if self.compression_enabled:
            pickled_data = zlib.compress(pickled_data)
        
        # Encrypt if enabled
        if self.encryption_enabled and self.fernet:
            pickled_data = self.fernet.encrypt(pickled_data)
        
        return pickled_data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        # Decrypt if enabled
        if self.encryption_enabled and self.fernet:
            data = self.fernet.decrypt(data)
        
        # Decompress if enabled
        if self.compression_enabled:
            data = zlib.decompress(data)
        
        # Unpickle the data
        return pickle.loads(data)

class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.metrics_buffer = []
        self.aggregated_metrics = {}
        
    async def record_metric(self, name: str, value: float, labels: Dict[str, str] = None,
                           timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        try:
            labels = labels or {}
            timestamp = timestamp or datetime.now()
            
            metric = {
                'name': name,
                'value': value,
                'labels': labels,
                'timestamp': timestamp.isoformat()
            }
            
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Cache the metric
            cache_key = f"metric:{name}:{self._hash_labels(labels)}"
            await self.cache_manager.set(cache_key, metric, ttl=86400)  # 24 hours
            
            # Trigger aggregation if buffer is full
            if len(self.metrics_buffer) >= 1000:
                await self._flush_metrics()
                
        except Exception as e:
            logger.error(f"Error recording metric: {e}")

    async def get_aggregated_metrics(self, metric_name: str, 
                                   aggregation: str = 'avg',
                                   time_range: timedelta = timedelta(hours=1)) -> Dict[str, float]:
        """Get aggregated metrics for a time range."""
        try:
            end_time = datetime.now()
            start_time = end_time - time_range
            
            # Get metrics from cache
            metrics = await self._get_metrics_in_range(metric_name, start_time, end_time)
            
            if not metrics:
                return {}
            
            # Aggregate metrics
            values = [m['value'] for m in metrics]
            
            aggregations = {
                'avg': np.mean(values),
                'sum': np.sum(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
                'std': np.std(values),
                'median': np.median(values)
            }
            
            if aggregation == 'all':
                return aggregations
            else:
                return {aggregation: aggregations.get(aggregation, 0.0)}
                
        except Exception as e:
            logger.error(f"Error getting aggregated metrics: {e}")
            return {}

    async def _get_metrics_in_range(self, metric_name: str, start_time: datetime, 
                                   end_time: datetime) -> List[Dict[str, Any]]:
        """Get metrics within a time range."""
        # This would typically query a time-series database
        # For now, return filtered buffer metrics
        filtered_metrics = []
        
        for metric in self.metrics_buffer:
            if metric['name'] == metric_name:
                metric_time = datetime.fromisoformat(metric['timestamp'])
                if start_time <= metric_time <= end_time:
                    filtered_metrics.append(metric)
        
        return filtered_metrics

    def _hash_labels(self, labels: Dict[str, str]) -> str:
        """Create a hash from labels dictionary."""
        label_str = json.dumps(labels, sort_keys=True)
        return hashlib.md5(label_str.encode()).hexdigest()

    async def _flush_metrics(self) -> None:
        """Flush metrics buffer to persistent storage."""
        try:
            # Here you would typically send metrics to a time-series database
            # For now, just clear the buffer
            logger.info(f"Flushed {len(self.metrics_buffer)} metrics")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

class PerformanceMonitor:
    """Advanced performance monitoring for analytics operations."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.active_operations = {}
        
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                start_time = datetime.now()
                operation_id = f"{operation_name}_{start_time.timestamp()}"
                
                try:
                    # Start monitoring
                    self.active_operations[operation_id] = {
                        'name': operation_name,
                        'start_time': start_time,
                        'status': 'running'
                    }
                    
                    # Execute function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Record success metrics
                    await self._record_operation_metrics(operation_id, True)
                    
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    await self._record_operation_metrics(operation_id, False, str(e))
                    raise
                finally:
                    # Clean up
                    if operation_id in self.active_operations:
                        del self.active_operations[operation_id]
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

    async def _record_operation_metrics(self, operation_id: str, success: bool, 
                                       error: Optional[str] = None) -> None:
        """Record performance metrics for an operation."""
        try:
            if operation_id not in self.active_operations:
                return
            
            operation = self.active_operations[operation_id]
            end_time = datetime.now()
            duration = (end_time - operation['start_time']).total_seconds()
            
            # Record duration metric
            await self.metrics_collector.record_metric(
                f"{operation['name']}_duration_seconds",
                duration,
                labels={'status': 'success' if success else 'error'}
            )
            
            # Record success/failure metric
            await self.metrics_collector.record_metric(
                f"{operation['name']}_total",
                1,
                labels={'status': 'success' if success else 'error'}
            )
            
            if not success and error:
                logger.error(f"Operation {operation['name']} failed: {error}")
            
        except Exception as e:
            logger.error(f"Error recording operation metrics: {e}")

class DataValidator:
    """Advanced data validation utilities."""
    
    def __init__(self):
        self.validation_rules = {}
        
    def add_validation_rule(self, name: str, rule: Callable[[Any], bool], 
                           error_message: str) -> None:
        """Add a custom validation rule."""
        self.validation_rules[name] = {
            'rule': rule,
            'error_message': error_message
        }
    
    async def validate_dataframe(self, df: pd.DataFrame, 
                                rules: List[str]) -> Tuple[bool, List[str]]:
        """Validate DataFrame against specified rules."""
        errors = []
        
        try:
            for rule_name in rules:
                if rule_name in self.validation_rules:
                    rule_info = self.validation_rules[rule_name]
                    
                    if not rule_info['rule'](df):
                        errors.append(rule_info['error_message'])
                else:
                    # Built-in rules
                    error = await self._apply_builtin_rule(df, rule_name)
                    if error:
                        errors.append(error)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False, [f"Validation error: {str(e)}"]

    async def _apply_builtin_rule(self, df: pd.DataFrame, rule_name: str) -> Optional[str]:
        """Apply built-in validation rules."""
        if rule_name == 'no_empty_dataframe':
            if df.empty:
                return "DataFrame cannot be empty"
        
        elif rule_name == 'no_all_null_columns':
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                return f"Columns with all null values: {', '.join(null_columns)}"
        
        elif rule_name == 'unique_primary_key':
            # Assumes first column is primary key
            if not df.empty and df.duplicated(subset=[df.columns[0]]).any():
                return "Primary key column contains duplicates"
        
        return None

# Utility functions
async def parallel_process(items: List[Any], processor: Callable[[Any], Any], 
                          max_workers: int = 4) -> List[Any]:
    """Process items in parallel using thread pool."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, processor, item)
            for item in items
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results

def create_data_hash(data: Union[pd.DataFrame, Dict, List]) -> str:
    """Create a hash of data for caching and comparison."""
    if isinstance(data, pd.DataFrame):
        data_str = data.to_string()
    elif isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive model performance metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score, log_loss
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr'))
            metrics['log_loss'] = float(log_loss(y_true, y_prob))
        except ValueError:
            # Handle cases where AUC ROC cannot be calculated
            pass
    
    return metrics

# Global instances
data_processor = DataProcessor()
cache_manager = CacheManager()
metrics_collector = MetricsCollector(cache_manager)
performance_monitor = PerformanceMonitor(metrics_collector)
data_validator = DataValidator()

# Add default validation rules
data_validator.add_validation_rule(
    'positive_values_only',
    lambda df: (df.select_dtypes(include=[np.number]) >= 0).all().all(),
    "DataFrame contains negative values"
)

data_validator.add_validation_rule(
    'no_infinite_values',
    lambda df: ~np.isinf(df.select_dtypes(include=[np.number])).any().any(),
    "DataFrame contains infinite values"
)

__all__ = [
    'DataProcessor',
    'CacheManager', 
    'MetricsCollector',
    'PerformanceMonitor',
    'DataValidator',
    'DataQualityReport',
    'PerformanceMetrics',
    'parallel_process',
    'create_data_hash',
    'format_bytes',
    'calculate_model_metrics',
    'data_processor',
    'cache_manager',
    'metrics_collector',
    'performance_monitor',
    'data_validator'
]
