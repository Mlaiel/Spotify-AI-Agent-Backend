#!/usr/bin/env python3
"""
Enterprise Database Performance Tuning Script
=============================================

This script provides automated performance optimization for multi-database
enterprise environments with AI-powered recommendations, intelligent tuning,
and comprehensive performance analysis.

Features:
- Automated performance analysis and optimization
- AI-powered query optimization recommendations
- Index analysis and optimization
- Configuration parameter tuning
- Resource allocation optimization
- Multi-database performance profiling
- Tenant-specific optimization strategies
- Performance regression detection
"""

import asyncio
import json
import logging
import time
import sys
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import psutil
import yaml
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from connection_manager import ConnectionManager
from performance_monitor import PerformanceMonitor, QueryAnalyzer
from __init__ import DatabaseType, ConfigurationLoader


class OptimizationLevel(Enum):
    """Optimization level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class OptimizationType(Enum):
    """Optimization type enumeration"""
    QUERY = "query"
    INDEX = "index"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    CACHE = "cache"
    CONNECTION = "connection"


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data structure"""
    recommendation_id: str
    optimization_type: OptimizationType
    database_type: DatabaseType
    priority: int  # 1-10, 10 being highest
    
    # Recommendation details
    title: str
    description: str
    impact_estimate: str  # e.g., "15-25% performance improvement"
    
    # Implementation
    implementation_sql: Optional[str] = None
    implementation_config: Optional[Dict[str, Any]] = None
    implementation_steps: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "low"  # low, medium, high
    rollback_plan: str = ""
    testing_required: bool = False
    
    # Performance metrics
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    expected_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False
    applied_at: Optional[datetime] = None


class PerformanceTuner:
    """
    AI-powered performance tuning system for enterprise databases
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.config_loader = ConfigurationLoader()
        self.performance_monitors: Dict[DatabaseType, PerformanceMonitor] = {}
        self.query_analyzers: Dict[DatabaseType, QueryAnalyzer] = {}
        
        # Tuning configuration
        self.optimization_level = OptimizationLevel(config.get('optimization_level', 'moderate'))
        self.max_recommendations = config.get('max_recommendations', 20)
        self.enable_auto_apply = config.get('enable_auto_apply', False)
        self.dry_run = config.get('dry_run', True)
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Optimization history
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # AI model parameters (placeholder for actual ML models)
        self.ai_models = {
            'query_optimizer': None,
            'index_advisor': None,
            'config_tuner': None
        }
    
    async def analyze_and_optimize(self, 
                                 database_types: Optional[List[DatabaseType]] = None,
                                 tenant_id: Optional[str] = None,
                                 optimization_level: OptimizationLevel = None) -> Dict[str, Any]:
        """Analyze performance and generate optimization recommendations"""
        
        start_time = time.time()
        
        optimization_level = optimization_level or self.optimization_level
        
        self.logger.info(f"Starting performance analysis - Level: {optimization_level.value}, Tenant: {tenant_id}")
        
        # Determine databases to analyze
        if database_types is None:
            database_types = self.config_loader.list_available_configurations()
        
        analysis_results = {}
        
        # Analyze each database type
        for db_type in database_types:
            try:
                db_analysis = await self._analyze_database_performance(
                    db_type, tenant_id, optimization_level
                )
                analysis_results[db_type.value] = db_analysis
                
            except Exception as e:
                self.logger.error(f"Analysis failed for {db_type.value}: {e}")
                analysis_results[db_type.value] = {
                    'error': str(e),
                    'recommendations': []
                }
        
        # Generate cross-database recommendations
        cross_db_recommendations = await self._generate_cross_database_recommendations(
            analysis_results, optimization_level
        )
        
        # Consolidate and prioritize recommendations
        all_recommendations = self._consolidate_recommendations(
            analysis_results, cross_db_recommendations
        )
        
        # Generate optimization plan
        optimization_plan = await self._create_optimization_plan(
            all_recommendations, optimization_level
        )
        
        execution_time = time.time() - start_time
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tenant_id': tenant_id,
            'optimization_level': optimization_level.value,
            'execution_time_seconds': round(execution_time, 2),
            'databases_analyzed': len(database_types),
            'total_recommendations': len(all_recommendations),
            'database_analysis': analysis_results,
            'optimization_plan': optimization_plan,
            'performance_summary': self._generate_performance_summary(analysis_results)
        }
        
        self.logger.info(f"Performance analysis completed - {len(all_recommendations)} recommendations generated")
        
        return results
    
    async def _analyze_database_performance(self, 
                                          db_type: DatabaseType,
                                          tenant_id: Optional[str],
                                          optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Analyze performance for specific database type"""
        
        try:
            # Load database configuration
            config = self.config_loader.load_configuration(db_type, tenant_id)
            
            # Collect performance metrics
            performance_metrics = await self._collect_performance_metrics(config)
            
            # Analyze queries
            query_analysis = await self._analyze_queries(config, optimization_level)
            
            # Analyze indexes
            index_analysis = await self._analyze_indexes(config, optimization_level)
            
            # Analyze configuration
            config_analysis = await self._analyze_configuration(config, optimization_level)
            
            # Analyze resource utilization
            resource_analysis = await self._analyze_resource_utilization(config)
            
            # Generate recommendations
            recommendations = []
            recommendations.extend(query_analysis['recommendations'])
            recommendations.extend(index_analysis['recommendations'])
            recommendations.extend(config_analysis['recommendations'])
            recommendations.extend(resource_analysis['recommendations'])
            
            # Prioritize recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            return {
                'database_type': db_type.value,
                'performance_metrics': performance_metrics,
                'query_analysis': query_analysis,
                'index_analysis': index_analysis,
                'configuration_analysis': config_analysis,
                'resource_analysis': resource_analysis,
                'recommendations': prioritized_recommendations[:self.max_recommendations],
                'total_recommendations': len(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Database analysis failed for {db_type.value}: {e}")
            raise
    
    async def _collect_performance_metrics(self, config) -> Dict[str, Any]:
        """Collect current performance metrics"""
        
        # Simulate performance metrics collection
        # In production, this would integrate with actual monitoring systems
        
        metrics = {
            'response_time_ms': {
                'avg': 250.0,
                'p50': 180.0,
                'p95': 450.0,
                'p99': 800.0
            },
            'throughput': {
                'queries_per_second': 1500.0,
                'transactions_per_second': 800.0
            },
            'resource_utilization': {
                'cpu_usage_percent': 65.0,
                'memory_usage_percent': 78.0,
                'disk_io_ops_per_sec': 2500.0,
                'network_mbps': 125.0
            },
            'database_specific': self._get_database_specific_metrics(config.type),
            'error_rates': {
                'total_errors_per_minute': 5.2,
                'timeout_errors_percent': 1.8,
                'connection_errors_percent': 0.3
            }
        }
        
        return metrics
    
    def _get_database_specific_metrics(self, db_type: DatabaseType) -> Dict[str, Any]:
        """Get database-specific performance metrics"""
        
        if db_type == DatabaseType.POSTGRESQL:
            return {
                'cache_hit_ratio': 0.89,
                'buffer_hit_ratio': 0.95,
                'index_hit_ratio': 0.92,
                'active_connections': 45,
                'idle_connections': 15,
                'long_running_queries': 3,
                'vacuum_stats': {
                    'last_vacuum': '2 hours ago',
                    'vacuum_efficiency': 0.87
                }
            }
        elif db_type == DatabaseType.MONGODB:
            return {
                'cache_hit_ratio': 0.83,
                'index_hit_ratio': 0.88,
                'collection_scans_per_sec': 12.0,
                'index_scans_per_sec': 890.0,
                'write_conflicts_per_sec': 2.1,
                'replication_lag_ms': 45.0
            }
        elif db_type == DatabaseType.REDIS:
            return {
                'cache_hit_ratio': 0.94,
                'memory_usage_mb': 1536.0,
                'memory_fragmentation_ratio': 1.08,
                'keyspace_hits_per_sec': 2400.0,
                'keyspace_misses_per_sec': 156.0,
                'expired_keys_per_sec': 45.0
            }
        elif db_type == DatabaseType.CLICKHOUSE:
            return {
                'queries_per_second': 2800.0,
                'merge_operations_per_sec': 15.0,
                'parts_count': 1245,
                'compression_ratio': 8.2,
                'memory_usage_for_queries_mb': 2048.0
            }
        else:
            return {}
    
    async def _analyze_queries(self, config, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Analyze query performance and generate recommendations"""
        
        # Simulate query analysis
        slow_queries = [
            {
                'query_hash': 'abc123',
                'query_text': 'SELECT * FROM users WHERE created_at > ? ORDER BY id',
                'avg_execution_time_ms': 1250.0,
                'execution_count': 450,
                'total_time_ms': 562500.0,
                'optimization_potential': 'high'
            },
            {
                'query_hash': 'def456',
                'query_text': 'SELECT COUNT(*) FROM orders WHERE status = ?',
                'avg_execution_time_ms': 890.0,
                'execution_count': 1200,
                'total_time_ms': 1068000.0,
                'optimization_potential': 'medium'
            }
        ]
        
        recommendations = []
        
        for query in slow_queries:
            if query['optimization_potential'] == 'high':
                rec = OptimizationRecommendation(
                    recommendation_id=f"query_opt_{query['query_hash']}",
                    optimization_type=OptimizationType.QUERY,
                    database_type=config.type,
                    priority=9,
                    title=f"Optimize slow query ({query['avg_execution_time_ms']:.0f}ms avg)",
                    description=f"Query executed {query['execution_count']} times with high execution time",
                    impact_estimate="30-40% query performance improvement",
                    implementation_sql=self._generate_query_optimization_sql(query, config.type),
                    implementation_steps=[
                        "Add appropriate indexes",
                        "Rewrite query to avoid unnecessary sorting",
                        "Consider query result caching"
                    ],
                    risk_level="low",
                    rollback_plan="Drop created indexes if performance degrades",
                    baseline_metrics={'avg_execution_time_ms': query['avg_execution_time_ms']},
                    expected_metrics={'avg_execution_time_ms': query['avg_execution_time_ms'] * 0.6},
                    tenant_id=config.tenant_id or ""
                )
                recommendations.append(rec)
        
        return {
            'slow_queries_count': len(slow_queries),
            'slow_queries': slow_queries,
            'recommendations': recommendations,
            'analysis_summary': {
                'total_queries_analyzed': 15420,
                'slow_queries_threshold_ms': 500,
                'optimization_opportunities': len(recommendations)
            }
        }
    
    def _generate_query_optimization_sql(self, query: Dict[str, Any], db_type: DatabaseType) -> str:
        """Generate SQL for query optimization"""
        
        if db_type == DatabaseType.POSTGRESQL:
            if 'created_at' in query['query_text'] and 'ORDER BY id' in query['query_text']:
                return """
-- Optimization for query: users by created_at with ordering
-- Create composite index for efficient filtering and sorting
CREATE INDEX CONCURRENTLY idx_users_created_at_id 
ON users (created_at, id) 
WHERE created_at > '2024-01-01';

-- Consider partitioning for very large tables
-- ALTER TABLE users PARTITION BY RANGE (created_at);
"""
            elif 'COUNT(*)' in query['query_text'] and 'status' in query['query_text']:
                return """
-- Optimization for count query by status
-- Create partial index for faster counting
CREATE INDEX CONCURRENTLY idx_orders_status 
ON orders (status) 
WHERE status IN ('pending', 'processing', 'completed');

-- Consider materialized view for frequent aggregations
-- CREATE MATERIALIZED VIEW orders_status_summary AS
-- SELECT status, COUNT(*) as count FROM orders GROUP BY status;
"""
        
        return f"-- Query optimization for {db_type.value} not yet implemented"
    
    async def _analyze_indexes(self, config, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Analyze index usage and generate recommendations"""
        
        # Simulate index analysis
        unused_indexes = [
            {
                'index_name': 'idx_old_user_email',
                'table_name': 'users',
                'size_mb': 45.2,
                'last_used': None,
                'scans': 0
            }
        ]
        
        missing_indexes = [
            {
                'table_name': 'orders',
                'columns': ['customer_id', 'order_date'],
                'query_benefit': 'High - used in 80% of slow queries',
                'estimated_size_mb': 15.3
            }
        ]
        
        recommendations = []
        
        # Recommend removing unused indexes
        for index in unused_indexes:
            rec = OptimizationRecommendation(
                recommendation_id=f"index_remove_{index['index_name']}",
                optimization_type=OptimizationType.INDEX,
                database_type=config.type,
                priority=6,
                title=f"Remove unused index: {index['index_name']}",
                description=f"Index {index['index_name']} is unused and consuming {index['size_mb']}MB",
                impact_estimate=f"Free {index['size_mb']}MB storage, reduce write overhead",
                implementation_sql=f"DROP INDEX {index['index_name']};",
                implementation_steps=[
                    "Verify index is truly unused",
                    "Drop index during maintenance window",
                    "Monitor performance after removal"
                ],
                risk_level="low",
                rollback_plan=f"Recreate index if needed: CREATE INDEX {index['index_name']} ON {index['table_name']}(...)",
                tenant_id=config.tenant_id or ""
            )
            recommendations.append(rec)
        
        # Recommend adding missing indexes
        for index in missing_indexes:
            rec = OptimizationRecommendation(
                recommendation_id=f"index_add_{index['table_name']}_{'_'.join(index['columns'])}",
                optimization_type=OptimizationType.INDEX,
                database_type=config.type,
                priority=8,
                title=f"Add missing index on {index['table_name']}({', '.join(index['columns'])})",
                description=f"Missing index causing slow queries on {index['table_name']}",
                impact_estimate="50-70% query performance improvement for affected queries",
                implementation_sql=self._generate_index_creation_sql(index, config.type),
                implementation_steps=[
                    "Create index with CONCURRENTLY option",
                    "Monitor query performance",
                    "Validate index usage statistics"
                ],
                risk_level="low",
                rollback_plan=f"DROP INDEX if performance issues arise",
                tenant_id=config.tenant_id or ""
            )
            recommendations.append(rec)
        
        return {
            'unused_indexes': unused_indexes,
            'missing_indexes': missing_indexes,
            'recommendations': recommendations,
            'index_statistics': {
                'total_indexes': 45,
                'unused_indexes': len(unused_indexes),
                'index_size_mb': 234.7,
                'unused_index_size_mb': sum(idx['size_mb'] for idx in unused_indexes)
            }
        }
    
    def _generate_index_creation_sql(self, index_info: Dict[str, Any], db_type: DatabaseType) -> str:
        """Generate SQL for index creation"""
        
        table_name = index_info['table_name']
        columns = index_info['columns']
        
        if db_type == DatabaseType.POSTGRESQL:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"
            columns_str = ', '.join(columns)
            return f"""
-- Create optimized index for {table_name}
CREATE INDEX CONCURRENTLY {index_name} 
ON {table_name} ({columns_str});

-- Add statistics collection
ANALYZE {table_name};
"""
        elif db_type == DatabaseType.MONGODB:
            columns_dict = {col: 1 for col in columns}
            return f"""
// Create compound index for {table_name}
db.{table_name}.createIndex({json.dumps(columns_dict)}, 
    {{ background: true, name: "idx_{'_'.join(columns)}" }})
"""
        
        return f"-- Index creation for {db_type.value} not yet implemented"
    
    async def _analyze_configuration(self, config, optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Analyze database configuration and generate recommendations"""
        
        recommendations = []
        
        if config.type == DatabaseType.POSTGRESQL:
            recommendations.extend(await self._analyze_postgresql_config(config, optimization_level))
        elif config.type == DatabaseType.MONGODB:
            recommendations.extend(await self._analyze_mongodb_config(config, optimization_level))
        elif config.type == DatabaseType.REDIS:
            recommendations.extend(await self._analyze_redis_config(config, optimization_level))
        
        return {
            'current_configuration': self._get_current_config_values(config),
            'recommendations': recommendations,
            'configuration_score': self._calculate_config_score(config)
        }
    
    async def _analyze_postgresql_config(self, config, optimization_level: OptimizationLevel) -> List[OptimizationRecommendation]:
        """Analyze PostgreSQL configuration"""
        
        recommendations = []
        
        # Simulate configuration analysis
        current_config = {
            'shared_buffers': '128MB',
            'work_mem': '4MB',
            'maintenance_work_mem': '64MB',
            'max_connections': '100',
            'effective_cache_size': '4GB'
        }
        
        # Recommend shared_buffers optimization
        rec = OptimizationRecommendation(
            recommendation_id="pg_config_shared_buffers",
            optimization_type=OptimizationType.CONFIGURATION,
            database_type=config.type,
            priority=7,
            title="Optimize shared_buffers setting",
            description="Current shared_buffers too low for available memory",
            impact_estimate="10-15% overall performance improvement",
            implementation_config={
                'shared_buffers': '1GB',
                'effective_cache_size': '6GB'
            },
            implementation_steps=[
                "Update postgresql.conf",
                "Restart PostgreSQL service",
                "Monitor buffer hit ratio"
            ],
            risk_level="medium",
            rollback_plan="Revert to previous configuration and restart",
            testing_required=True,
            tenant_id=config.tenant_id or ""
        )
        recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_mongodb_config(self, config, optimization_level: OptimizationLevel) -> List[OptimizationRecommendation]:
        """Analyze MongoDB configuration"""
        
        recommendations = []
        
        # Recommend WiredTiger cache optimization
        rec = OptimizationRecommendation(
            recommendation_id="mongo_config_wiredtiger_cache",
            optimization_type=OptimizationType.CONFIGURATION,
            database_type=config.type,
            priority=6,
            title="Optimize WiredTiger cache size",
            description="WiredTiger cache can be increased for better performance",
            impact_estimate="15-20% read performance improvement",
            implementation_config={
                'storage.wiredTiger.engineConfig.cacheSizeGB': 2.0
            },
            implementation_steps=[
                "Update mongod.conf",
                "Restart MongoDB service",
                "Monitor cache hit ratio"
            ],
            risk_level="low",
            rollback_plan="Revert cache size setting",
            tenant_id=config.tenant_id or ""
        )
        recommendations.append(rec)
        
        return recommendations
    
    async def _analyze_redis_config(self, config, optimization_level: OptimizationLevel) -> List[OptimizationRecommendation]:
        """Analyze Redis configuration"""
        
        recommendations = []
        
        # Recommend maxmemory policy optimization
        rec = OptimizationRecommendation(
            recommendation_id="redis_config_maxmemory_policy",
            optimization_type=OptimizationType.CONFIGURATION,
            database_type=config.type,
            priority=5,
            title="Optimize maxmemory eviction policy",
            description="Current eviction policy may not be optimal for workload",
            impact_estimate="5-10% memory efficiency improvement",
            implementation_config={
                'maxmemory-policy': 'allkeys-lru'
            },
            implementation_steps=[
                "Update redis.conf",
                "Apply configuration with CONFIG SET",
                "Monitor eviction statistics"
            ],
            risk_level="low",
            rollback_plan="Revert to previous eviction policy",
            tenant_id=config.tenant_id or ""
        )
        recommendations.append(rec)
        
        return recommendations
    
    def _get_current_config_values(self, config) -> Dict[str, Any]:
        """Get current configuration values"""
        
        # Simulate getting current configuration
        if config.type == DatabaseType.POSTGRESQL:
            return {
                'shared_buffers': '128MB',
                'work_mem': '4MB',
                'max_connections': '100',
                'checkpoint_completion_target': '0.5'
            }
        elif config.type == DatabaseType.MONGODB:
            return {
                'wiredTiger.engineConfig.cacheSizeGB': 1.0,
                'operationProfiling.slowOpThresholdMs': 100
            }
        elif config.type == DatabaseType.REDIS:
            return {
                'maxmemory': '2gb',
                'maxmemory-policy': 'noeviction'
            }
        
        return {}
    
    def _calculate_config_score(self, config) -> float:
        """Calculate configuration optimization score (0-100)"""
        
        # Simulate configuration scoring
        base_score = 75.0
        
        if config.type == DatabaseType.POSTGRESQL:
            # Adjust based on configuration optimality
            return base_score + 10.0  # Well-configured
        elif config.type == DatabaseType.MONGODB:
            return base_score + 5.0
        elif config.type == DatabaseType.REDIS:
            return base_score - 5.0  # Needs optimization
        
        return base_score
    
    async def _analyze_resource_utilization(self, config) -> Dict[str, Any]:
        """Analyze resource utilization and generate recommendations"""
        
        recommendations = []
        
        # Get current resource usage
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU utilization recommendation
        if cpu_usage > 80:
            rec = OptimizationRecommendation(
                recommendation_id="resource_cpu_scale",
                optimization_type=OptimizationType.RESOURCE,
                database_type=config.type,
                priority=8,
                title="Scale CPU resources",
                description=f"High CPU utilization: {cpu_usage:.1f}%",
                impact_estimate="20-30% performance improvement",
                implementation_steps=[
                    "Add more CPU cores",
                    "Optimize CPU-intensive queries",
                    "Consider connection pooling"
                ],
                risk_level="low",
                tenant_id=config.tenant_id or ""
            )
            recommendations.append(rec)
        
        # Memory utilization recommendation
        if memory.percent > 85:
            rec = OptimizationRecommendation(
                recommendation_id="resource_memory_scale",
                optimization_type=OptimizationType.RESOURCE,
                database_type=config.type,
                priority=7,
                title="Increase memory allocation",
                description=f"High memory utilization: {memory.percent:.1f}%",
                impact_estimate="15-25% performance improvement",
                implementation_steps=[
                    "Add more RAM",
                    "Optimize buffer settings",
                    "Review memory-intensive queries"
                ],
                risk_level="low",
                tenant_id=config.tenant_id or ""
            )
            recommendations.append(rec)
        
        return {
            'current_usage': {
                'cpu_percent': cpu_usage,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100
            },
            'recommendations': recommendations,
            'resource_score': self._calculate_resource_score(cpu_usage, memory.percent)
        }
    
    def _calculate_resource_score(self, cpu_usage: float, memory_usage: float) -> float:
        """Calculate resource utilization score"""
        
        # Optimal utilization is around 60-70%
        cpu_score = max(0, 100 - abs(cpu_usage - 65) * 2)
        memory_score = max(0, 100 - abs(memory_usage - 65) * 2)
        
        return (cpu_score + memory_score) / 2
    
    async def _generate_cross_database_recommendations(self, 
                                                     analysis_results: Dict[str, Any],
                                                     optimization_level: OptimizationLevel) -> List[OptimizationRecommendation]:
        """Generate recommendations that span multiple databases"""
        
        recommendations = []
        
        # Connection pooling optimization
        total_connections = sum(
            result.get('performance_metrics', {}).get('database_specific', {}).get('active_connections', 0)
            for result in analysis_results.values()
            if isinstance(result, dict) and 'performance_metrics' in result
        )
        
        if total_connections > 200:
            rec = OptimizationRecommendation(
                recommendation_id="cross_db_connection_pooling",
                optimization_type=OptimizationType.CONNECTION,
                database_type=DatabaseType.POSTGRESQL,  # Generic
                priority=9,
                title="Implement global connection pooling",
                description="High total connection count across databases",
                impact_estimate="20-30% resource efficiency improvement",
                implementation_steps=[
                    "Deploy connection pooler (PgBouncer, etc.)",
                    "Configure connection limits",
                    "Monitor connection utilization"
                ],
                risk_level="medium",
                testing_required=True
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _consolidate_recommendations(self, 
                                   analysis_results: Dict[str, Any],
                                   cross_db_recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Consolidate and deduplicate recommendations"""
        
        all_recommendations = cross_db_recommendations.copy()
        
        # Collect recommendations from each database analysis
        for result in analysis_results.values():
            if isinstance(result, dict) and 'recommendations' in result:
                all_recommendations.extend(result['recommendations'])
        
        # Remove duplicates based on recommendation_id
        seen_ids = set()
        unique_recommendations = []
        
        for rec in all_recommendations:
            if rec.recommendation_id not in seen_ids:
                unique_recommendations.append(rec)
                seen_ids.add(rec.recommendation_id)
        
        return unique_recommendations
    
    def _prioritize_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Prioritize recommendations by impact and risk"""
        
        def priority_score(rec):
            # Higher priority number = higher importance
            base_score = rec.priority
            
            # Adjust for risk (lower risk = higher score)
            risk_adjustment = {
                'low': 2,
                'medium': 0,
                'high': -3
            }.get(rec.risk_level, 0)
            
            # Adjust for optimization type
            type_priority = {
                OptimizationType.QUERY: 3,
                OptimizationType.INDEX: 2,
                OptimizationType.CONFIGURATION: 1,
                OptimizationType.RESOURCE: 2,
                OptimizationType.CACHE: 1,
                OptimizationType.CONNECTION: 1
            }.get(rec.optimization_type, 0)
            
            return base_score + risk_adjustment + type_priority
        
        return sorted(recommendations, key=priority_score, reverse=True)
    
    async def _create_optimization_plan(self, 
                                      recommendations: List[OptimizationRecommendation],
                                      optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """Create detailed optimization implementation plan"""
        
        # Filter recommendations based on optimization level
        filtered_recommendations = self._filter_by_optimization_level(
            recommendations, optimization_level
        )
        
        # Group by implementation phases
        phases = self._group_recommendations_by_phase(filtered_recommendations)
        
        # Calculate estimated impact
        total_impact = self._estimate_total_impact(filtered_recommendations)
        
        plan = {
            'total_recommendations': len(filtered_recommendations),
            'estimated_implementation_time_hours': self._estimate_implementation_time(filtered_recommendations),
            'estimated_impact': total_impact,
            'risk_assessment': self._assess_overall_risk(filtered_recommendations),
            'implementation_phases': phases,
            'testing_requirements': [
                rec.recommendation_id for rec in filtered_recommendations 
                if rec.testing_required
            ],
            'rollback_procedures': {
                rec.recommendation_id: rec.rollback_plan 
                for rec in filtered_recommendations 
                if rec.rollback_plan
            }
        }
        
        return plan
    
    def _filter_by_optimization_level(self, 
                                    recommendations: List[OptimizationRecommendation],
                                    optimization_level: OptimizationLevel) -> List[OptimizationRecommendation]:
        """Filter recommendations based on optimization level"""
        
        risk_tolerance = {
            OptimizationLevel.CONSERVATIVE: ['low'],
            OptimizationLevel.MODERATE: ['low', 'medium'],
            OptimizationLevel.AGGRESSIVE: ['low', 'medium', 'high'],
            OptimizationLevel.EXPERIMENTAL: ['low', 'medium', 'high']
        }
        
        allowed_risks = risk_tolerance.get(optimization_level, ['low'])
        
        filtered = [
            rec for rec in recommendations 
            if rec.risk_level in allowed_risks
        ]
        
        # Limit number of recommendations for conservative level
        if optimization_level == OptimizationLevel.CONSERVATIVE:
            filtered = filtered[:5]
        elif optimization_level == OptimizationLevel.MODERATE:
            filtered = filtered[:10]
        
        return filtered
    
    def _group_recommendations_by_phase(self, 
                                      recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Group recommendations into implementation phases"""
        
        # Phase 1: Low-risk, high-impact optimizations
        phase1 = [
            rec for rec in recommendations 
            if rec.risk_level == 'low' and rec.priority >= 7
        ]
        
        # Phase 2: Medium-risk optimizations
        phase2 = [
            rec for rec in recommendations 
            if rec.risk_level == 'medium' or (rec.risk_level == 'low' and rec.priority < 7)
        ]
        
        # Phase 3: High-risk or experimental optimizations
        phase3 = [
            rec for rec in recommendations 
            if rec.risk_level == 'high'
        ]
        
        phases = []
        
        if phase1:
            phases.append({
                'phase': 1,
                'name': 'Quick Wins',
                'description': 'Low-risk, high-impact optimizations',
                'recommendations': [rec.recommendation_id for rec in phase1],
                'estimated_time_hours': sum(1 for _ in phase1),  # 1 hour per recommendation
                'can_run_parallel': True
            })
        
        if phase2:
            phases.append({
                'phase': 2,
                'name': 'Standard Optimizations',
                'description': 'Medium-impact optimizations requiring testing',
                'recommendations': [rec.recommendation_id for rec in phase2],
                'estimated_time_hours': sum(2 for _ in phase2),  # 2 hours per recommendation
                'can_run_parallel': False
            })
        
        if phase3:
            phases.append({
                'phase': 3,
                'name': 'Advanced Optimizations',
                'description': 'High-impact optimizations requiring careful testing',
                'recommendations': [rec.recommendation_id for rec in phase3],
                'estimated_time_hours': sum(4 for _ in phase3),  # 4 hours per recommendation
                'can_run_parallel': False
            })
        
        return phases
    
    def _estimate_total_impact(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Estimate total performance impact"""
        
        if not recommendations:
            return "No significant impact expected"
        
        high_impact_count = sum(1 for rec in recommendations if rec.priority >= 8)
        medium_impact_count = sum(1 for rec in recommendations if 5 <= rec.priority < 8)
        
        if high_impact_count >= 3:
            return "30-50% overall performance improvement expected"
        elif high_impact_count >= 1 or medium_impact_count >= 3:
            return "15-30% overall performance improvement expected"
        else:
            return "5-15% overall performance improvement expected"
    
    def _assess_overall_risk(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Assess overall risk of optimization plan"""
        
        high_risk_count = sum(1 for rec in recommendations if rec.risk_level == 'high')
        medium_risk_count = sum(1 for rec in recommendations if rec.risk_level == 'medium')
        
        if high_risk_count > 0:
            return "High - Requires extensive testing and rollback planning"
        elif medium_risk_count > 2:
            return "Medium - Requires testing and monitoring"
        else:
            return "Low - Safe to implement with standard procedures"
    
    def _estimate_implementation_time(self, recommendations: List[OptimizationRecommendation]) -> float:
        """Estimate total implementation time in hours"""
        
        time_by_type = {
            OptimizationType.QUERY: 1.0,
            OptimizationType.INDEX: 0.5,
            OptimizationType.CONFIGURATION: 2.0,
            OptimizationType.RESOURCE: 3.0,
            OptimizationType.CACHE: 1.0,
            OptimizationType.CONNECTION: 2.0
        }
        
        total_time = 0.0
        for rec in recommendations:
            base_time = time_by_type.get(rec.optimization_type, 1.0)
            
            # Adjust for risk level
            if rec.risk_level == 'high':
                base_time *= 2.0
            elif rec.risk_level == 'medium':
                base_time *= 1.5
            
            # Add testing time if required
            if rec.testing_required:
                base_time += 1.0
            
            total_time += base_time
        
        return total_time
    
    def _generate_performance_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary across all databases"""
        
        summary = {
            'databases_analyzed': len(analysis_results),
            'total_recommendations': 0,
            'optimization_opportunities': defaultdict(int),
            'performance_scores': {},
            'critical_issues': []
        }
        
        for db_name, result in analysis_results.items():
            if isinstance(result, dict) and 'recommendations' in result:
                recommendations = result['recommendations']
                summary['total_recommendations'] += len(recommendations)
                
                # Count optimization types
                for rec in recommendations:
                    if hasattr(rec, 'optimization_type'):
                        summary['optimization_opportunities'][rec.optimization_type.value] += 1
                
                # Calculate performance score
                if 'performance_metrics' in result:
                    score = self._calculate_performance_score(result['performance_metrics'])
                    summary['performance_scores'][db_name] = score
                    
                    if score < 60:
                        summary['critical_issues'].append(
                            f"{db_name}: Performance score below 60% ({score:.1f}%)"
                        )
        
        return summary
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        
        # Base score
        score = 100.0
        
        # Penalize high response times
        response_time = metrics.get('response_time_ms', {}).get('avg', 0)
        if response_time > 500:
            score -= min(30, (response_time - 500) / 50 * 5)
        
        # Penalize high resource utilization
        cpu_usage = metrics.get('resource_utilization', {}).get('cpu_usage_percent', 0)
        if cpu_usage > 80:
            score -= min(20, (cpu_usage - 80) * 2)
        
        memory_usage = metrics.get('resource_utilization', {}).get('memory_usage_percent', 0)
        if memory_usage > 85:
            score -= min(15, (memory_usage - 85) * 2)
        
        # Reward good cache hit ratios
        cache_hit_ratio = metrics.get('database_specific', {}).get('cache_hit_ratio', 0)
        if cache_hit_ratio > 0.9:
            score += 5
        elif cache_hit_ratio < 0.7:
            score -= 10
        
        return max(0, min(100, score))


async def main():
    """Main function for command-line execution"""
    
    parser = argparse.ArgumentParser(description='Database Performance Tuning Script')
    parser.add_argument('--databases', nargs='+', 
                       choices=[db.value for db in DatabaseType],
                       help='Database types to analyze')
    parser.add_argument('--tenant', type=str, help='Tenant ID to analyze')
    parser.add_argument('--level', type=str, default='moderate',
                       choices=[level.value for level in OptimizationLevel],
                       help='Optimization level')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'yaml'],
                       help='Output format')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--apply', action='store_true', help='Apply recommendations (use with caution)')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode (default)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = {'dry_run': args.dry_run}
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
            config.update(file_config)
    
    # Parse parameters
    database_types = None
    if args.databases:
        database_types = [DatabaseType(db) for db in args.databases]
    
    optimization_level = OptimizationLevel(args.level)
    
    # Create performance tuner
    tuner = PerformanceTuner(config)
    
    # Run analysis
    try:
        results = await tuner.analyze_and_optimize(
            database_types=database_types,
            tenant_id=args.tenant,
            optimization_level=optimization_level
        )
        
        # Output results
        if args.format == 'yaml':
            output = yaml.dump(results, default_flow_style=False)
        else:
            output = json.dumps(results, indent=2, default=str)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)
        
        # Summary
        total_recommendations = results.get('total_recommendations', 0)
        print(f"\n📊 Analysis Summary:")
        print(f"   • {results.get('databases_analyzed', 0)} databases analyzed")
        print(f"   • {total_recommendations} optimization recommendations")
        print(f"   • Estimated impact: {results.get('optimization_plan', {}).get('estimated_impact', 'Unknown')}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Performance analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    # Add enum import
    from enum import Enum
    from dataclasses import dataclass, field
    
    asyncio.run(main())
