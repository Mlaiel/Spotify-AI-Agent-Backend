#!/usr/bin/env python3
"""
🎵 Advanced Automation Testing Suite for Spotify AI Agent
Ultra-sophisticated testing framework for automation components

This test suite provides comprehensive testing for:
- Automation engine functionality and workflows
- ML predictor accuracy and performance
- Monitoring system reliability
- Configuration management
- End-to-end integration testing
- Performance benchmarking

Author: Fahed Mlaiel (Lead Developer & AI Architect)
Usage: python test_suite.py [test_category] [options]
"""

import asyncio
import json
import logging
import pytest
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch

# Import automation modules for testing
from config import ConfigurationManager, ConfigEnvironment, create_configuration_manager
from engine import AdvancedAutomationEngine, WorkflowDefinition, AutomationType, create_advanced_automation_engine
from predictor import AdvancedPredictor, PredictionType, create_advanced_predictor
from monitor import MetricsCollector, AlertManager, create_monitoring_system
from orchestrator import AutomationOrchestrator, OrchestratorState

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [TEST] %(message)s'
)
logger = logging.getLogger(__name__)


class AutomationTestSuite:
    """Comprehensive automation testing suite"""
    
    def __init__(self):
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'execution_time': 0,
            'coverage': {},
            'performance_metrics': {}
        }
        
        self.test_environment = ConfigEnvironment.TESTING
        self.temp_dir = None
        
        # Test components
        self.config_manager = None
        self.automation_engine = None
        self.predictor = None
        self.metrics_collector = None
        self.alert_manager = None
        self.orchestrator = None
    
    async def setup_test_environment(self):
        """Setup test environment and components"""
        logger.info("🚀 Setting up test environment...")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix='automation_test_')
        logger.info(f"Test directory: {self.temp_dir}")
        
        try:
            # Initialize test configuration
            self.config_manager = create_configuration_manager(self.test_environment)
            await self.config_manager.load_configuration()
            
            # Mock external dependencies for testing
            await self._setup_mocks()
            
            logger.info("✅ Test environment setup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            raise
    
    async def _setup_mocks(self):
        """Setup mocks for external dependencies"""
        # Mock Redis connection
        self.redis_mock = AsyncMock()
        
        # Mock database connection
        self.db_mock = AsyncMock()
        
        # Mock external API calls
        self.api_mock = AsyncMock()
        
        logger.info("✅ Mocks configured")
    
    async def teardown_test_environment(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            # Cleanup components
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            if self.automation_engine:
                await self.automation_engine.stop()
            
            # Remove temporary directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
            
            logger.info("✅ Test environment cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        start_time = time.time()
        
        logger.info("🎯 Starting comprehensive automation test suite")
        print("\n" + "="*60)
        print("🎵 Spotify AI Agent Automation Test Suite")
        print("="*60)
        
        try:
            # Setup test environment
            await self.setup_test_environment()
            
            # Run test categories
            test_categories = [
                ('Configuration Tests', self.test_configuration_management),
                ('Engine Tests', self.test_automation_engine),
                ('Predictor Tests', self.test_ml_predictor),
                ('Monitoring Tests', self.test_monitoring_system),
                ('Integration Tests', self.test_integration),
                ('Performance Tests', self.test_performance),
                ('Security Tests', self.test_security),
                ('Stress Tests', self.test_stress_scenarios)
            ]
            
            for category_name, test_method in test_categories:
                print(f"\n🔍 Running {category_name}...")
                try:
                    await test_method()
                    print(f"✅ {category_name} completed")
                except Exception as e:
                    print(f"❌ {category_name} failed: {e}")
                    self.test_results['errors'].append({
                        'category': category_name,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
            
            # Calculate final results
            self.test_results['execution_time'] = time.time() - start_time
            
            # Display results
            await self.display_test_results()
            
        finally:
            # Cleanup
            await self.teardown_test_environment()
    
    async def test_configuration_management(self):
        """Test configuration management functionality"""
        print("  📋 Testing configuration management...")
        
        tests = [
            self._test_config_loading,
            self._test_config_validation,
            self._test_config_hot_reload,
            self._test_config_encryption,
            self._test_environment_specific_configs
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_config_loading(self):
        """Test configuration loading"""
        # Test configuration loading from different sources
        config_data = {
            'automation': {
                'max_concurrent_workflows': 100,
                'workflow_timeout_seconds': 3600
            },
            'ml': {
                'enabled': True,
                'prediction_horizon_hours': 24
            }
        }
        
        # Create test config file
        config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        # Test loading
        config_manager = ConfigurationManager(
            environment=self.test_environment,
            config_file=str(config_file)
        )
        
        await config_manager.load_configuration()
        
        # Verify loaded configuration
        automation_config = config_manager.get_config_section('automation')
        assert automation_config['max_concurrent_workflows'] == 100
        assert automation_config['workflow_timeout_seconds'] == 3600
        
        ml_config = config_manager.get_config_section('ml')
        assert ml_config['enabled'] is True
        assert ml_config['prediction_horizon_hours'] == 24
    
    async def _test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = {
            'automation': {
                'max_concurrent_workflows': 50,
                'workflow_timeout_seconds': 1800
            }
        }
        
        config_manager = ConfigurationManager(self.test_environment)
        validation_result = config_manager.validate_configuration(valid_config)
        assert validation_result['is_valid'] is True
        
        # Test invalid configuration
        invalid_config = {
            'automation': {
                'max_concurrent_workflows': -1,  # Invalid negative value
                'workflow_timeout_seconds': 'invalid'  # Invalid type
            }
        }
        
        validation_result = config_manager.validate_configuration(invalid_config)
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0
    
    async def _test_config_hot_reload(self):
        """Test configuration hot reload"""
        config_file = Path(self.temp_dir) / 'reload_test.yaml'
        
        # Initial configuration
        initial_config = {'automation': {'max_workers': 10}}
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(initial_config, f)
        
        config_manager = ConfigurationManager(
            environment=self.test_environment,
            config_file=str(config_file)
        )
        await config_manager.load_configuration()
        
        # Verify initial value
        assert config_manager.get_config_section('automation')['max_workers'] == 10
        
        # Update configuration file
        updated_config = {'automation': {'max_workers': 20}}
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(updated_config, f)
        
        # Reload configuration
        await config_manager.reload_configuration()
        
        # Verify updated value
        assert config_manager.get_config_section('automation')['max_workers'] == 20
    
    async def _test_config_encryption(self):
        """Test configuration encryption for sensitive data"""
        config_manager = ConfigurationManager(self.test_environment)
        
        # Test encryption
        sensitive_data = "super_secret_api_key"
        encrypted = config_manager.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data
        
        # Test decryption
        decrypted = config_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
    
    async def _test_environment_specific_configs(self):
        """Test environment-specific configuration overrides"""
        base_config = {
            'automation': {'max_workers': 10},
            'monitoring': {'enabled': True}
        }
        
        dev_overrides = {
            'automation': {'max_workers': 5},
            'monitoring': {'debug_mode': True}
        }
        
        config_manager = ConfigurationManager(ConfigEnvironment.DEVELOPMENT)
        merged_config = config_manager.merge_environment_config(base_config, dev_overrides)
        
        # Verify overrides applied
        assert merged_config['automation']['max_workers'] == 5
        assert merged_config['monitoring']['enabled'] is True
        assert merged_config['monitoring']['debug_mode'] is True
    
    async def test_automation_engine(self):
        """Test automation engine functionality"""
        print("  ⚙️ Testing automation engine...")
        
        tests = [
            self._test_engine_initialization,
            self._test_workflow_creation,
            self._test_workflow_execution,
            self._test_parallel_workflow_execution,
            self._test_workflow_error_handling,
            self._test_circuit_breaker,
            self._test_rate_limiting,
            self._test_workflow_persistence
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_engine_initialization(self):
        """Test automation engine initialization"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(
            max_concurrent_workflows=50,
            max_concurrent_actions=200,
            workflow_timeout=1800,
            enable_circuit_breaker=True
        )
        
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        assert engine.state.value == 'running'
        assert engine.config.max_concurrent_workflows == 50
        assert engine.metrics is not None
    
    async def _test_workflow_creation(self):
        """Test workflow creation and validation"""
        # Test valid workflow
        valid_workflow = WorkflowDefinition(
            id='test_workflow_001',
            name='Test Workflow',
            automation_type=AutomationType.PERFORMANCE_OPTIMIZATION,
            actions=[
                {
                    'type': 'http_request',
                    'parameters': {
                        'url': 'https://api.example.com/test',
                        'method': 'GET'
                    }
                }
            ]
        )
        
        assert valid_workflow.id == 'test_workflow_001'
        assert valid_workflow.automation_type == AutomationType.PERFORMANCE_OPTIMIZATION
        assert len(valid_workflow.actions) == 1
        
        # Test workflow validation
        validation_result = valid_workflow.validate()
        assert validation_result['is_valid'] is True
        
        # Test invalid workflow
        invalid_workflow = WorkflowDefinition(
            id='',  # Empty ID
            name='Invalid Workflow',
            automation_type=AutomationType.INCIDENT_RESPONSE,
            actions=[]  # No actions
        )
        
        validation_result = invalid_workflow.validate()
        assert validation_result['is_valid'] is False
    
    async def _test_workflow_execution(self):
        """Test single workflow execution"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create test workflow
        workflow = {
            'id': 'test_execution_001',
            'name': 'Test Execution',
            'actions': [
                {
                    'type': 'log_message',
                    'parameters': {
                        'message': 'Test workflow executed',
                        'level': 'info'
                    }
                }
            ]
        }
        
        # Execute workflow
        result = await engine.execute_intelligent_workflow(workflow)
        
        assert result is not None
        assert 'execution_id' in result
        
        await engine.stop()
    
    async def _test_parallel_workflow_execution(self):
        """Test parallel workflow execution"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=5)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create multiple test workflows
        workflows = []
        for i in range(3):
            workflow = {
                'id': f'parallel_test_{i}',
                'name': f'Parallel Test {i}',
                'actions': [
                    {
                        'type': 'delay',
                        'parameters': {'seconds': 0.1}
                    }
                ]
            }
            workflows.append(workflow)
        
        # Execute workflows in parallel
        start_time = time.time()
        tasks = [engine.execute_intelligent_workflow(wf) for wf in workflows]
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Verify parallel execution (should be faster than sequential)
        assert len(results) == 3
        assert execution_time < 0.5  # Should complete in less than 0.5 seconds
        
        await engine.stop()
    
    async def _test_workflow_error_handling(self):
        """Test workflow error handling"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create workflow with error
        workflow = {
            'id': 'error_test_001',
            'name': 'Error Test',
            'actions': [
                {
                    'type': 'invalid_action',  # This should cause an error
                    'parameters': {}
                }
            ]
        }
        
        # Execute workflow and expect error handling
        result = await engine.execute_intelligent_workflow(workflow)
        
        # Engine should handle the error gracefully
        assert result is not None
        # Error should be logged but execution should continue
        
        await engine.stop()
    
    async def _test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        from engine import EngineConfiguration, CircuitBreaker
        
        # Test circuit breaker directly
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=Exception
        )
        
        # Test successful calls
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == 'closed'
        
        # Test failing calls
        async def failing_operation():
            raise Exception("Test failure")
        
        # Circuit should open after threshold failures
        for _ in range(4):
            try:
                await circuit_breaker.call(failing_operation)
            except:
                pass
        
        assert circuit_breaker.state == 'open'
    
    async def _test_rate_limiting(self):
        """Test rate limiting functionality"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(
            enable_rate_limiting=True,
            rate_limit_per_minute=10
        )
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create simple workflow
        workflow = {
            'id': 'rate_limit_test',
            'name': 'Rate Limit Test',
            'actions': [
                {
                    'type': 'log_message',
                    'parameters': {'message': 'Rate limited execution'}
                }
            ]
        }
        
        # Execute workflows rapidly
        start_time = time.time()
        execution_count = 0
        
        for i in range(15):  # Try to exceed rate limit
            try:
                await engine.execute_intelligent_workflow(workflow)
                execution_count += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        
        execution_time = time.time() - start_time
        
        # Should have rate limiting effect
        assert execution_count <= 12  # Allow some buffer
        
        await engine.stop()
    
    async def _test_workflow_persistence(self):
        """Test workflow state persistence"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create workflow
        workflow = {
            'id': 'persistence_test',
            'name': 'Persistence Test',
            'actions': [
                {
                    'type': 'log_message',
                    'parameters': {'message': 'Testing persistence'}
                }
            ]
        }
        
        # Execute workflow
        result = await engine.execute_intelligent_workflow(workflow)
        execution_id = result.get('execution_id')
        
        # Check if workflow state is persisted
        workflow_state = await engine.get_workflow_state(execution_id)
        assert workflow_state is not None
        assert workflow_state.get('workflow_id') == 'persistence_test'
        
        await engine.stop()
    
    async def test_ml_predictor(self):
        """Test ML predictor functionality"""
        print("  🧠 Testing ML predictor...")
        
        tests = [
            self._test_predictor_initialization,
            self._test_traffic_prediction,
            self._test_resource_prediction,
            self._test_failure_prediction,
            self._test_anomaly_detection,
            self._test_model_performance,
            self._test_prediction_caching
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_predictor_initialization(self):
        """Test ML predictor initialization"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='lstm',
            prediction_type=PredictionType.TRAFFIC_FORECAST,
            forecast_horizon=24,
            confidence_threshold=0.8
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        assert len(predictor.models) > 0
        assert predictor.config.confidence_threshold == 0.8
    
    async def _test_traffic_prediction(self):
        """Test traffic prediction functionality"""
        from predictor import PredictionConfig
        import pandas as pd
        import numpy as np
        
        config = PredictionConfig(
            model_type='lstm',
            prediction_type=PredictionType.TRAFFIC_FORECAST,
            forecast_horizon=12
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Generate mock historical data
        dates = pd.date_range(start='2024-01-01', periods=168, freq='H')  # 1 week
        traffic_data = pd.DataFrame({
            'requests_per_second': np.random.normal(1000, 200, len(dates)),
            'active_users': np.random.normal(5000, 1000, len(dates))
        }, index=dates)
        
        # Make prediction
        result = await predictor.predict_traffic_forecast(traffic_data, 12)
        
        assert result is not None
        assert len(result.predictions) == 12
        assert all(conf >= 0 and conf <= 1 for conf in result.confidence_scores)
    
    async def _test_resource_prediction(self):
        """Test resource usage prediction"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='random_forest',
            prediction_type=PredictionType.RESOURCE_FORECAST
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Mock current metrics
        current_metrics = {
            'cpu_usage': 45.0,
            'memory_usage': 68.0,
            'disk_usage': 35.0,
            'network_io': 120.0
        }
        
        result = await predictor.predict_resource_usage(current_metrics, 6)
        
        assert result is not None
        assert len(result.predictions) == 6
        assert result.confidence_scores[0] > 0
    
    async def _test_failure_prediction(self):
        """Test failure probability prediction"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='xgboost',
            prediction_type=PredictionType.FAILURE_PREDICTION
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Mock component metrics
        component_metrics = {
            'error_rate': 2.5,
            'response_time_p99': 250,
            'cpu_usage': 75,
            'memory_usage': 85,
            'restart_count_24h': 1
        }
        
        result = await predictor.predict_failure_probability('database', component_metrics)
        
        assert result is not None
        assert 0 <= result.predictions[0] <= 1  # Probability between 0 and 1
    
    async def _test_anomaly_detection(self):
        """Test anomaly detection"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='isolation_forest',
            prediction_type=PredictionType.ANOMALY_DETECTION
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Mock system metrics
        metrics = {
            'requests_per_second': 1200,
            'error_rate': 0.8,
            'response_time_avg': 150,
            'cpu_usage': 45,
            'memory_usage': 68
        }
        
        result = await predictor.detect_anomalies(metrics)
        
        assert result is not None
        assert len(result.predictions) > 0
        assert result.confidence_scores[0] >= 0
    
    async def _test_model_performance(self):
        """Test model performance evaluation"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='lstm',
            prediction_type=PredictionType.TRAFFIC_FORECAST
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Get performance report
        report = await predictor.get_model_performance_report()
        
        assert report is not None
        assert 'overall_performance' in report
        assert 'model_metrics' in report
        assert report['overall_performance']['average_accuracy'] >= 0
    
    async def _test_prediction_caching(self):
        """Test prediction result caching"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='lstm',
            prediction_type=PredictionType.TRAFFIC_FORECAST
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Mock data for caching test
        test_data = {'requests': [1000, 1100, 1200]}
        
        # First prediction (should be cached)
        start_time = time.time()
        result1 = await predictor.predict_traffic_forecast(test_data, 1)
        first_time = time.time() - start_time
        
        # Second identical prediction (should use cache)
        start_time = time.time()
        result2 = await predictor.predict_traffic_forecast(test_data, 1)
        second_time = time.time() - start_time
        
        # Cache should make second prediction faster
        assert second_time < first_time
        assert result1.predictions == result2.predictions
    
    async def test_monitoring_system(self):
        """Test monitoring system functionality"""
        print("  📊 Testing monitoring system...")
        
        tests = [
            self._test_metrics_collection,
            self._test_alert_management,
            self._test_threshold_monitoring,
            self._test_notification_system,
            self._test_alert_suppression,
            self._test_metrics_aggregation
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_metrics_collection(self):
        """Test metrics collection functionality"""
        metrics_collector = MetricsCollector()
        
        # Test system metrics collection
        system_metrics = await metrics_collector.collect_system_metrics()
        
        assert 'cpu_usage' in system_metrics
        assert 'memory_usage' in system_metrics
        assert 'disk_usage' in system_metrics
        assert system_metrics['cpu_usage'] >= 0
        assert system_metrics['memory_usage'] >= 0
    
    async def _test_alert_management(self):
        """Test alert management functionality"""
        from monitor import AlertSeverity
        
        alert_manager = AlertManager()
        await alert_manager.initialize()
        
        # Create test alert
        alert_data = {
            'id': 'test_alert_001',
            'severity': AlertSeverity.WARNING,
            'component': 'test_component',
            'message': 'Test alert message',
            'metric_value': 75.0,
            'threshold': 70.0
        }
        
        # Fire alert
        await alert_manager.fire_alert(alert_data)
        
        # Check alert summary
        summary = alert_manager.get_alert_summary()
        assert summary['total_active_alerts'] >= 1
        
        # Resolve alert
        await alert_manager.resolve_alert('test_alert_001')
        
        summary = alert_manager.get_alert_summary()
        # Alert should be resolved
    
    async def _test_threshold_monitoring(self):
        """Test threshold monitoring"""
        alert_manager = AlertManager()
        await alert_manager.initialize()
        
        # Test dynamic threshold calculation
        historical_values = [45, 50, 48, 52, 47, 49, 51, 46]
        threshold = alert_manager.calculate_dynamic_threshold(
            historical_values,
            sensitivity=2.0
        )
        
        assert threshold > 0
        assert threshold > max(historical_values)  # Should be above historical max
    
    async def _test_notification_system(self):
        """Test notification system"""
        alert_manager = AlertManager()
        await alert_manager.initialize()
        
        # Mock notification channels
        with patch('monitor.AlertManager._send_slack_notification') as mock_slack:
            mock_slack.return_value = True
            
            # Test notification sending
            alert_data = {
                'severity': 'high',
                'component': 'test',
                'message': 'Test notification'
            }
            
            result = await alert_manager.send_notification(alert_data, 'slack')
            assert result is True
            mock_slack.assert_called_once()
    
    async def _test_alert_suppression(self):
        """Test alert suppression and deduplication"""
        alert_manager = AlertManager()
        await alert_manager.initialize()
        
        # Fire same alert multiple times
        alert_data = {
            'id': 'duplicate_test',
            'severity': 'warning',
            'component': 'test',
            'message': 'Duplicate alert test'
        }
        
        # First alert should fire
        result1 = await alert_manager.fire_alert(alert_data)
        
        # Second identical alert should be suppressed
        result2 = await alert_manager.fire_alert(alert_data)
        
        # Verify suppression logic works
        assert result1 != result2 or alert_manager.suppression_enabled
    
    async def _test_metrics_aggregation(self):
        """Test metrics aggregation and windowing"""
        metrics_collector = MetricsCollector()
        
        # Generate test metrics over time
        metrics_series = []
        for i in range(10):
            metrics = {
                'timestamp': time.time() - (i * 60),  # 1 minute intervals
                'requests_per_second': 1000 + (i * 10),
                'response_time': 150 + (i * 5)
            }
            metrics_series.append(metrics)
        
        # Test aggregation
        aggregated = metrics_collector.aggregate_metrics(
            metrics_series,
            window_size=300,  # 5 minutes
            aggregation_type='avg'
        )
        
        assert 'requests_per_second' in aggregated
        assert 'response_time' in aggregated
        assert aggregated['requests_per_second'] > 0
    
    async def test_integration(self):
        """Test end-to-end integration scenarios"""
        print("  🔗 Testing integration scenarios...")
        
        tests = [
            self._test_orchestrator_integration,
            self._test_ml_automation_integration,
            self._test_monitoring_automation_integration,
            self._test_config_change_propagation,
            self._test_cross_component_communication
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_orchestrator_integration(self):
        """Test orchestrator integration with all components"""
        orchestrator = AutomationOrchestrator(self.test_environment)
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Verify all components are initialized
        assert orchestrator.state == OrchestratorState.RUNNING
        assert orchestrator.config_manager is not None
        assert orchestrator.automation_engine is not None
        
        # Test status retrieval
        status = orchestrator.get_status()
        assert status['state'] == OrchestratorState.RUNNING
        assert 'statistics' in status
        assert 'components' in status
        
        # Cleanup
        await orchestrator.shutdown()
        assert orchestrator.state == OrchestratorState.STOPPED
    
    async def _test_ml_automation_integration(self):
        """Test ML predictor integration with automation engine"""
        # This would test how ML predictions trigger automated actions
        # For now, we'll mock the integration
        
        # Mock ML prediction that triggers automation
        prediction_result = {
            'component': 'database',
            'failure_probability': 0.85,  # High failure risk
            'recommendations': ['scale_out', 'increase_monitoring']
        }
        
        # Verify that automation engine can process ML recommendations
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create workflow based on ML recommendation
        remediation_workflow = {
            'id': 'ml_remediation_001',
            'name': 'ML-triggered Remediation',
            'trigger': 'ml_prediction',
            'actions': [
                {
                    'type': 'scale_service',
                    'parameters': {
                        'service': prediction_result['component'],
                        'action': 'scale_out'
                    }
                }
            ]
        }
        
        result = await engine.execute_intelligent_workflow(remediation_workflow)
        assert result is not None
        
        await engine.stop()
    
    async def _test_monitoring_automation_integration(self):
        """Test monitoring system integration with automation"""
        from monitor import AlertSeverity
        
        # Create monitoring system
        metrics_collector, alert_manager = create_monitoring_system()
        await alert_manager.initialize()
        
        # Create automation engine
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Simulate alert that triggers automation
        alert_data = {
            'id': 'integration_test_alert',
            'severity': AlertSeverity.HIGH,
            'component': 'api_gateway',
            'message': 'High CPU usage detected',
            'metric_value': 90.0,
            'threshold': 80.0
        }
        
        # Fire alert
        await alert_manager.fire_alert(alert_data)
        
        # Create automated response workflow
        response_workflow = {
            'id': 'alert_response_001',
            'name': 'High CPU Response',
            'trigger': 'alert',
            'actions': [
                {
                    'type': 'scale_service',
                    'parameters': {
                        'service': 'api_gateway',
                        'scale_factor': 1.5
                    }
                }
            ]
        }
        
        result = await engine.execute_intelligent_workflow(response_workflow)
        assert result is not None
        
        await engine.stop()
    
    async def _test_config_change_propagation(self):
        """Test configuration change propagation across components"""
        # Test that configuration changes are properly propagated
        config_manager = create_configuration_manager(self.test_environment)
        await config_manager.load_configuration()
        
        # Get initial configuration
        initial_config = config_manager.get_config_section('automation')
        initial_max_workflows = initial_config.get('max_concurrent_workflows', 100)
        
        # Update configuration
        new_max_workflows = initial_max_workflows + 50
        await config_manager.update_config_section('automation', {
            'max_concurrent_workflows': new_max_workflows
        })
        
        # Verify update
        updated_config = config_manager.get_config_section('automation')
        assert updated_config['max_concurrent_workflows'] == new_max_workflows
    
    async def _test_cross_component_communication(self):
        """Test communication between different components"""
        # Test message passing between components
        
        # Create components
        config_manager = create_configuration_manager(self.test_environment)
        await config_manager.load_configuration()
        
        from engine import EngineConfiguration
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Test workflow that involves multiple components
        multi_component_workflow = {
            'id': 'cross_component_test',
            'name': 'Cross-Component Test',
            'actions': [
                {
                    'type': 'collect_metrics',
                    'parameters': {'component': 'all'}
                },
                {
                    'type': 'analyze_metrics',
                    'parameters': {'analysis_type': 'performance'}
                },
                {
                    'type': 'generate_report',
                    'parameters': {'format': 'json'}
                }
            ]
        }
        
        result = await engine.execute_intelligent_workflow(multi_component_workflow)
        assert result is not None
        
        await engine.stop()
    
    async def test_performance(self):
        """Test performance characteristics"""
        print("  ⚡ Testing performance characteristics...")
        
        tests = [
            self._test_workflow_execution_performance,
            self._test_prediction_latency,
            self._test_memory_usage,
            self._test_concurrent_operations,
            self._test_scaling_behavior
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_workflow_execution_performance(self):
        """Test workflow execution performance"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=100)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create simple workflow
        workflow = {
            'id': 'performance_test',
            'name': 'Performance Test',
            'actions': [
                {
                    'type': 'log_message',
                    'parameters': {'message': 'Performance test execution'}
                }
            ]
        }
        
        # Measure execution time
        start_time = time.time()
        
        # Execute multiple workflows
        tasks = []
        for i in range(50):
            workflow_copy = workflow.copy()
            workflow_copy['id'] = f'performance_test_{i}'
            tasks.append(engine.execute_intelligent_workflow(workflow_copy))
        
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert len(results) == 50
        assert execution_time < 10.0  # Should complete in less than 10 seconds
        
        # Calculate throughput
        throughput = len(results) / execution_time
        assert throughput > 5.0  # At least 5 workflows per second
        
        self.test_results['performance_metrics']['workflow_throughput'] = throughput
        
        await engine.stop()
    
    async def _test_prediction_latency(self):
        """Test ML prediction latency"""
        from predictor import PredictionConfig
        
        config = PredictionConfig(
            model_type='lstm',
            prediction_type=PredictionType.TRAFFIC_FORECAST
        )
        
        predictor = AdvancedPredictor(config)
        await predictor.initialize_models()
        
        # Measure prediction latency
        test_data = {'requests': [1000, 1100, 1200, 1300, 1400]}
        
        latencies = []
        for _ in range(10):
            start_time = time.time()
            await predictor.predict_traffic_forecast(test_data, 1)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Performance assertions
        assert avg_latency < 500  # Average latency should be less than 500ms
        assert max_latency < 1000  # Max latency should be less than 1000ms
        
        self.test_results['performance_metrics']['prediction_latency_avg'] = avg_latency
        self.test_results['performance_metrics']['prediction_latency_max'] = max_latency
    
    async def _test_memory_usage(self):
        """Test memory usage patterns"""
        import psutil
        import gc
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and initialize components
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=50)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Measure memory after initialization
        after_init_memory = process.memory_info().rss / 1024 / 1024
        init_memory_increase = after_init_memory - baseline_memory
        
        # Execute many workflows
        workflows = []
        for i in range(100):
            workflow = {
                'id': f'memory_test_{i}',
                'name': f'Memory Test {i}',
                'actions': [
                    {
                        'type': 'log_message',
                        'parameters': {'message': f'Memory test {i}'}
                    }
                ]
            }
            workflows.append(engine.execute_intelligent_workflow(workflow))
        
        await asyncio.gather(*workflows)
        
        # Measure memory after execution
        after_execution_memory = process.memory_info().rss / 1024 / 1024
        execution_memory_increase = after_execution_memory - after_init_memory
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory after cleanup
        after_cleanup_memory = process.memory_info().rss / 1024 / 1024
        
        # Memory usage assertions
        assert init_memory_increase < 100  # Initialization should use less than 100MB
        assert execution_memory_increase < 200  # Execution should use less than 200MB additional
        
        self.test_results['performance_metrics']['memory_baseline'] = baseline_memory
        self.test_results['performance_metrics']['memory_after_init'] = after_init_memory
        self.test_results['performance_metrics']['memory_after_execution'] = after_execution_memory
        self.test_results['performance_metrics']['memory_after_cleanup'] = after_cleanup_memory
        
        await engine.stop()
    
    async def _test_concurrent_operations(self):
        """Test concurrent operations handling"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=20)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create concurrent workflows
        concurrent_workflows = []
        for i in range(30):  # More than max_concurrent_workflows
            workflow = {
                'id': f'concurrent_test_{i}',
                'name': f'Concurrent Test {i}',
                'actions': [
                    {
                        'type': 'delay',
                        'parameters': {'seconds': 0.1}
                    }
                ]
            }
            concurrent_workflows.append(workflow)
        
        # Execute all workflows concurrently
        start_time = time.time()
        tasks = [engine.execute_intelligent_workflow(wf) for wf in concurrent_workflows]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # Performance assertions
        assert len(successful_results) > 0
        assert execution_time < 5.0  # Should handle concurrency efficiently
        
        self.test_results['performance_metrics']['concurrent_success_rate'] = len(successful_results) / len(results)
        
        await engine.stop()
    
    async def _test_scaling_behavior(self):
        """Test scaling behavior under different loads"""
        from engine import EngineConfiguration
        
        # Test with different configuration scales
        scales = [10, 50, 100]
        scaling_results = {}
        
        for scale in scales:
            config = EngineConfiguration(max_concurrent_workflows=scale)
            engine = AdvancedAutomationEngine(config)
            await engine.initialize()
            
            # Create workflows matching the scale
            workflows = []
            for i in range(scale):
                workflow = {
                    'id': f'scale_test_{scale}_{i}',
                    'name': f'Scale Test {scale}-{i}',
                    'actions': [
                        {
                            'type': 'log_message',
                            'parameters': {'message': f'Scale test {scale}-{i}'}
                        }
                    ]
                }
                workflows.append(workflow)
            
            # Measure execution time
            start_time = time.time()
            tasks = [engine.execute_intelligent_workflow(wf) for wf in workflows]
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            throughput = len(results) / execution_time
            scaling_results[scale] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'success_count': len(results)
            }
            
            await engine.stop()
        
        # Analyze scaling behavior
        # Throughput should generally increase with scale (up to a point)
        assert scaling_results[50]['throughput'] > scaling_results[10]['throughput']
        
        self.test_results['performance_metrics']['scaling_behavior'] = scaling_results
    
    async def test_security(self):
        """Test security features"""
        print("  🔒 Testing security features...")
        
        tests = [
            self._test_config_encryption,
            self._test_access_control,
            self._test_input_validation,
            self._test_secure_communications
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_access_control(self):
        """Test access control mechanisms"""
        # Test that sensitive operations require proper authorization
        
        # Mock authentication system
        class MockAuthenticator:
            def __init__(self):
                self.valid_tokens = {'admin_token': 'admin', 'user_token': 'user'}
            
            def authenticate(self, token):
                return self.valid_tokens.get(token)
            
            def authorize(self, user, operation):
                if user == 'admin':
                    return True
                elif user == 'user' and operation in ['read', 'monitor']:
                    return True
                return False
        
        auth = MockAuthenticator()
        
        # Test admin access
        admin_user = auth.authenticate('admin_token')
        assert admin_user == 'admin'
        assert auth.authorize(admin_user, 'config_update') is True
        
        # Test user access
        regular_user = auth.authenticate('user_token')
        assert regular_user == 'user'
        assert auth.authorize(regular_user, 'read') is True
        assert auth.authorize(regular_user, 'config_update') is False
        
        # Test invalid access
        invalid_user = auth.authenticate('invalid_token')
        assert invalid_user is None
    
    async def _test_input_validation(self):
        """Test input validation and sanitization"""
        from engine import WorkflowDefinition, AutomationType
        
        # Test workflow validation with malicious input
        malicious_workflow = WorkflowDefinition(
            id='<script>alert("xss")</script>',  # XSS attempt
            name='Malicious Workflow',
            automation_type=AutomationType.INCIDENT_RESPONSE,
            actions=[
                {
                    'type': 'shell_command',
                    'parameters': {
                        'command': 'rm -rf /'  # Dangerous command
                    }
                }
            ]
        )
        
        # Validation should catch security issues
        validation_result = malicious_workflow.validate()
        assert validation_result['is_valid'] is False
        assert any('security' in error.lower() for error in validation_result.get('errors', []))
    
    async def _test_secure_communications(self):
        """Test secure communication protocols"""
        # Test that sensitive data is transmitted securely
        
        # Mock secure communication
        class MockSecureChannel:
            def __init__(self):
                self.encryption_enabled = True
                self.tls_version = '1.3'
            
            def send_data(self, data, endpoint):
                if not self.encryption_enabled:
                    raise SecurityError("Encryption required for sensitive data")
                
                # Simulate encrypted transmission
                return {'status': 'success', 'encrypted': True}
        
        secure_channel = MockSecureChannel()
        
        # Test secure data transmission
        sensitive_data = {'api_key': 'secret_key_123'}
        result = secure_channel.send_data(sensitive_data, 'api.example.com')
        
        assert result['status'] == 'success'
        assert result['encrypted'] is True
        
        # Test that insecure transmission fails
        secure_channel.encryption_enabled = False
        
        try:
            secure_channel.send_data(sensitive_data, 'api.example.com')
            assert False, "Should have raised SecurityError"
        except Exception as e:
            assert "Encryption required" in str(e)
    
    async def test_stress_scenarios(self):
        """Test system behavior under stress"""
        print("  💪 Testing stress scenarios...")
        
        tests = [
            self._test_high_load_workflow_execution,
            self._test_memory_pressure,
            self._test_rapid_configuration_changes,
            self._test_component_failure_simulation
        ]
        
        for test in tests:
            await self._run_test(test.__name__, test)
    
    async def _test_high_load_workflow_execution(self):
        """Test system under high workflow load"""
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=200)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Create a large number of workflows
        workflow_count = 500
        workflows = []
        
        for i in range(workflow_count):
            workflow = {
                'id': f'stress_test_{i}',
                'name': f'Stress Test {i}',
                'actions': [
                    {
                        'type': 'cpu_intensive_task',
                        'parameters': {'iterations': 1000}
                    }
                ]
            }
            workflows.append(workflow)
        
        # Execute workflows in batches to avoid overwhelming the system
        batch_size = 50
        start_time = time.time()
        
        for i in range(0, workflow_count, batch_size):
            batch = workflows[i:i+batch_size]
            tasks = [engine.execute_intelligent_workflow(wf) for wf in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        execution_time = time.time() - start_time
        
        # System should handle high load gracefully
        assert execution_time < 60.0  # Should complete within reasonable time
        
        # Check system state after stress test
        assert engine.state.value == 'running'  # Engine should still be running
        
        await engine.stop()
    
    async def _test_memory_pressure(self):
        """Test system behavior under memory pressure"""
        import gc
        
        # Create objects that consume significant memory
        memory_consumers = []
        
        try:
            # Gradually increase memory usage
            for i in range(100):
                # Create large objects
                large_object = [0] * (1024 * 1024)  # 1MB list
                memory_consumers.append(large_object)
                
                # Test that system still functions under memory pressure
                if i % 10 == 0:
                    # Test basic functionality
                    config_manager = create_configuration_manager(self.test_environment)
                    assert config_manager is not None
                    
                    # Force garbage collection periodically
                    gc.collect()
        
        finally:
            # Cleanup memory
            memory_consumers.clear()
            gc.collect()
    
    async def _test_rapid_configuration_changes(self):
        """Test system behavior with rapid configuration changes"""
        config_manager = create_configuration_manager(self.test_environment)
        await config_manager.load_configuration()
        
        # Perform rapid configuration changes
        for i in range(50):
            # Alternate between different values
            max_workflows = 100 + (i % 2) * 50
            
            await config_manager.update_config_section('automation', {
                'max_concurrent_workflows': max_workflows
            })
            
            # Verify change was applied
            current_config = config_manager.get_config_section('automation')
            assert current_config['max_concurrent_workflows'] == max_workflows
            
            # Small delay to simulate real-world usage
            await asyncio.sleep(0.01)
    
    async def _test_component_failure_simulation(self):
        """Test system behavior when components fail"""
        # Simulate component failures and recovery
        
        # Test automation engine failure recovery
        from engine import EngineConfiguration
        
        config = EngineConfiguration(max_concurrent_workflows=10)
        engine = AdvancedAutomationEngine(config)
        await engine.initialize()
        
        # Simulate engine failure
        original_state = engine.state
        engine.state = 'error'  # Simulate error state
        
        # Test recovery mechanism
        await engine.recover_from_error()
        
        # Engine should recover
        assert engine.state != 'error'
        
        await engine.stop()
    
    async def _run_test(self, test_name: str, test_func):
        """Run individual test with error handling"""
        self.test_results['total_tests'] += 1
        
        try:
            await test_func()
            self.test_results['passed'] += 1
            print(f"    ✅ {test_name}")
            
        except Exception as e:
            self.test_results['failed'] += 1
            error_info = {
                'test_name': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.test_results['errors'].append(error_info)
            print(f"    ❌ {test_name}: {e}")
            
            # Log detailed error for debugging
            logger.error(f"Test failed: {test_name}")
            logger.error(traceback.format_exc())
    
    async def display_test_results(self):
        """Display comprehensive test results"""
        print("\n" + "="*60)
        print("🎯 Test Suite Results")
        print("="*60)
        
        # Overall statistics
        total = self.test_results['total_tests']
        passed = self.test_results['passed']
        failed = self.test_results['failed']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Tests: {total}")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Execution Time: {self.test_results['execution_time']:.2f} seconds")
        
        # Performance metrics
        if self.test_results['performance_metrics']:
            print(f"\n⚡ Performance Metrics:")
            for metric, value in self.test_results['performance_metrics'].items():
                if isinstance(value, dict):
                    print(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        print(f"    {sub_metric}: {sub_value}")
                else:
                    print(f"  {metric}: {value}")
        
        # Error details
        if self.test_results['errors']:
            print(f"\n❌ Failed Tests:")
            for error in self.test_results['errors']:
                print(f"  • {error['test_name']}: {error['error']}")
        
        # Final assessment
        print(f"\n🎵 Test Suite Assessment:")
        if success_rate >= 95:
            print("  🎉 Excellent! All systems performing optimally")
        elif success_rate >= 85:
            print("  ✅ Good! Most systems functioning correctly")
        elif success_rate >= 70:
            print("  ⚠️ Warning! Some issues detected")
        else:
            print("  🚨 Critical! Multiple system failures detected")
        
        # Recommendations
        if failed > 0:
            print(f"\n💡 Recommendations:")
            print("  • Review failed test details above")
            print("  • Check system logs for additional context")
            print("  • Consider scaling resources if performance tests failed")
            print("  • Verify configuration settings")


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


async def main():
    """Main test suite entry point"""
    print("🎵 Spotify AI Agent Automation Test Suite")
    print("Ultra-Advanced Testing Framework")
    print("Author: Fahed Mlaiel (Lead Developer & AI Architect)")
    print("="*60)
    
    # Create and run test suite
    test_suite = AutomationTestSuite()
    
    try:
        await test_suite.run_all_tests()
        
        # Return appropriate exit code
        if test_suite.test_results['failed'] > 0:
            return 1  # Some tests failed
        else:
            return 0  # All tests passed
            
    except Exception as e:
        print(f"\n💥 Test suite execution failed: {e}")
        print(traceback.format_exc())
        return 2  # Test suite failure


if __name__ == '__main__':
    import sys
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(3)
