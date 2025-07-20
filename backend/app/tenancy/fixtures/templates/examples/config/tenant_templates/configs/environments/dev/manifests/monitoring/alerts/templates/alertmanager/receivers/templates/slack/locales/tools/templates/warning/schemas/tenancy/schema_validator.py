#!/usr/bin/env python3
"""
Schema Validation & Testing Framework
=====================================

Framework complet de validation et tests pour les schémas tenancy
avec couverture de tests, benchmarks de performance et validation ML.
"""

import json
import time
import asyncio
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import unittest
import pytest
from pydantic import ValidationError
import logging

# Import des schémas pour validation
from .tenant_config_schema import TenantConfigSchema, TenantType, TenantStatus
from .alert_schema import AlertSchema, TenantAlertSchema, AlertSeverity
from .warning_schema import WarningSchema, TenantWarningSchema, WarningSeverity
from .notification_schema import NotificationSchema, NotificationChannel
from .monitoring_schema import MonitoringConfigSchema, MonitoringMetric
from .compliance_schema import ComplianceSchema, ComplianceStandard
from .performance_schema import PerformanceMetricsSchema, PerformanceBaseline
from .schema_factory import SchemaFactory, SchemaBuilder, SchemaBuilderConfig


@dataclass
class ValidationResult:
    """Résultat d'une validation de schéma."""
    schema_name: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0
    schema_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark de performance."""
    operation_name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_ops_per_sec: float


class SchemaValidator:
    """Validateur de schémas avec support des règles métier."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Charge les règles de validation métier."""
        return {
            "tenant_config": {
                "required_fields": ["tenant_id", "tenant_name", "admin_email"],
                "tenant_id_pattern": r"^[a-z0-9_]{3,50}$",
                "email_domain_whitelist": [],  # Vide = tous autorisés
                "max_tenant_name_length": 100,
                "min_sla_uptime": 99.0
            },
            "alerts": {
                "max_conditions_per_alert": 10,
                "required_notification_channels": 1,
                "max_recipients": 20,
                "allowed_severities": ["low", "medium", "high", "critical"],
                "escalation_required_for": ["critical"]
            },
            "monitoring": {
                "max_metrics_per_config": 100,
                "min_collection_interval_seconds": 10,
                "max_retention_days": 365,
                "required_metric_types": ["counter", "gauge", "histogram"]
            },
            "compliance": {
                "min_controls_per_standard": {
                    "gdpr": 5,
                    "soc2": 3,
                    "hipaa": 4,
                    "iso27001": 10
                },
                "required_retention_policies": ["user_data", "logs", "backups"]
            }
        }
    
    def validate_tenant_config(self, config_data: Dict[str, Any]) -> ValidationResult:
        """Valide une configuration de tenant."""
        start_time = time.time()
        result = ValidationResult(schema_name="tenant_config", schema_data=config_data)
        
        try:
            # Validation Pydantic
            tenant_config = TenantConfigSchema(**config_data)
            
            # Validation des règles métier
            rules = self.validation_rules["tenant_config"]
            
            # Vérifier les champs requis
            missing_fields = [field for field in rules["required_fields"] 
                            if field not in config_data or not config_data[field]]
            if missing_fields:
                result.errors.append(f"Missing required fields: {missing_fields}")
            
            # Vérifier la longueur du nom de tenant
            if len(config_data.get("tenant_name", "")) > rules["max_tenant_name_length"]:
                result.errors.append(
                    f"Tenant name exceeds maximum length of {rules['max_tenant_name_length']}"
                )
            
            # Vérifier le SLA minimum
            sla_data = config_data.get("sla", {})
            if isinstance(sla_data, dict):
                uptime = sla_data.get("uptime_percentage", 0)
                if uptime < rules["min_sla_uptime"]:
                    result.warnings.append(
                        f"SLA uptime ({uptime}%) below recommended minimum ({rules['min_sla_uptime']}%)"
                    )
            
            result.is_valid = len(result.errors) == 0
            
        except ValidationError as e:
            result.is_valid = False
            result.errors.extend([str(error) for error in e.errors()])
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected validation error: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result
    
    def validate_alert_config(self, alert_data: Dict[str, Any]) -> ValidationResult:
        """Valide une configuration d'alerte."""
        start_time = time.time()
        result = ValidationResult(schema_name="alert_config", schema_data=alert_data)
        
        try:
            # Validation Pydantic
            alert_config = AlertSchema(**alert_data)
            
            # Validation des règles métier
            rules = self.validation_rules["alerts"]
            
            # Vérifier le nombre de conditions
            conditions = alert_data.get("conditions", [])
            if len(conditions) > rules["max_conditions_per_alert"]:
                result.errors.append(
                    f"Too many conditions ({len(conditions)}), maximum allowed: {rules['max_conditions_per_alert']}"
                )
            
            # Vérifier les canaux de notification
            channels = alert_data.get("notification_channels", [])
            if len(channels) < rules["required_notification_channels"]:
                result.errors.append(
                    f"At least {rules['required_notification_channels']} notification channel required"
                )
            
            # Vérifier l'escalation pour les alertes critiques
            severity = alert_data.get("severity", "").lower()
            if severity in rules["escalation_required_for"]:
                escalation = alert_data.get("escalation_policy")
                if not escalation or not escalation.get("enabled", False):
                    result.warnings.append(
                        f"Escalation policy recommended for {severity} alerts"
                    )
            
            result.is_valid = len(result.errors) == 0
            
        except ValidationError as e:
            result.is_valid = False
            result.errors.extend([str(error) for error in e.errors()])
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected validation error: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result
    
    def validate_monitoring_config(self, monitoring_data: Dict[str, Any]) -> ValidationResult:
        """Valide une configuration de monitoring."""
        start_time = time.time()
        result = ValidationResult(schema_name="monitoring_config", schema_data=monitoring_data)
        
        try:
            # Validation Pydantic
            monitoring_config = MonitoringConfigSchema(**monitoring_data)
            
            # Validation des règles métier
            rules = self.validation_rules["monitoring"]
            
            # Vérifier le nombre de métriques
            metrics = monitoring_data.get("metrics", [])
            if len(metrics) > rules["max_metrics_per_config"]:
                result.errors.append(
                    f"Too many metrics ({len(metrics)}), maximum allowed: {rules['max_metrics_per_config']}"
                )
            
            # Vérifier l'intervalle de collecte
            interval = monitoring_data.get("collection_interval_seconds", 0)
            if interval < rules["min_collection_interval_seconds"]:
                result.errors.append(
                    f"Collection interval too low ({interval}s), minimum: {rules['min_collection_interval_seconds']}s"
                )
            
            # Vérifier la rétention
            retention = monitoring_data.get("retention_days", 0)
            if retention > rules["max_retention_days"]:
                result.warnings.append(
                    f"Long retention period ({retention} days) may impact storage costs"
                )
            
            result.is_valid = len(result.errors) == 0
            
        except ValidationError as e:
            result.is_valid = False
            result.errors.extend([str(error) for error in e.errors()])
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected validation error: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result
    
    def validate_compliance_config(self, compliance_data: Dict[str, Any]) -> ValidationResult:
        """Valide une configuration de compliance."""
        start_time = time.time()
        result = ValidationResult(schema_name="compliance_config", schema_data=compliance_data)
        
        try:
            # Validation Pydantic
            compliance_config = ComplianceSchema(**compliance_data)
            
            # Validation des règles métier
            rules = self.validation_rules["compliance"]
            
            # Vérifier les contrôles par standard
            standards = compliance_data.get("applicable_standards", [])
            controls = compliance_data.get("controls", [])
            
            for standard in standards:
                standard_name = standard.lower() if isinstance(standard, str) else standard.value.lower()
                if standard_name in rules["min_controls_per_standard"]:
                    min_controls = rules["min_controls_per_standard"][standard_name]
                    standard_controls = [c for c in controls 
                                       if standard_name in [s.lower() if isinstance(s, str) else s.value.lower() 
                                                          for s in c.get("standards", [])]]
                    
                    if len(standard_controls) < min_controls:
                        result.warnings.append(
                            f"Only {len(standard_controls)} controls for {standard_name}, "
                            f"recommended minimum: {min_controls}"
                        )
            
            result.is_valid = len(result.errors) == 0
            
        except ValidationError as e:
            result.is_valid = False
            result.errors.extend([str(error) for error in e.errors()])
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected validation error: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result


class SchemaBenchmark:
    """Framework de benchmark pour les schémas."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def benchmark_schema_creation(self, schema_class, test_data: Dict[str, Any], 
                                iterations: int = 1000) -> BenchmarkResult:
        """Benchmark la création d'instances de schéma."""
        
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                schema_class(**test_data)
            except Exception:
                pass  # Ignorer les erreurs pour le benchmark
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convertir en ms
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / (total_time / 1000)  # ops/sec
        
        return BenchmarkResult(
            operation_name=f"{schema_class.__name__} creation",
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput
        )
    
    def benchmark_schema_validation(self, validator: SchemaValidator, 
                                  validation_method: Callable, test_data: Dict[str, Any],
                                  iterations: int = 500) -> BenchmarkResult:
        """Benchmark la validation de schémas."""
        
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            validation_method(test_data)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / (total_time / 1000)
        
        return BenchmarkResult(
            operation_name=f"{validation_method.__name__} validation",
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput
        )
    
    async def benchmark_factory_creation(self, iterations: int = 100) -> List[BenchmarkResult]:
        """Benchmark la création complète via factory."""
        
        results = []
        
        # Configuration de test
        config = SchemaBuilderConfig(
            tenant_type=TenantType.PROFESSIONAL,
            compliance_standards=[ComplianceStandard.GDPR],
            auto_optimize=True
        )
        
        builder = SchemaBuilder(config)
        
        # Benchmark création complète
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            await builder.build_complete_tenant(
                tenant_id=f"test_tenant_{_}",
                tenant_name=f"Test Tenant {_}",
                admin_email=f"admin{_}@test.com"
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / (total_time / 1000)
        
        results.append(BenchmarkResult(
            operation_name="Complete tenant creation",
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput
        ))
        
        return results


class SchemaTestSuite:
    """Suite de tests complète pour les schémas."""
    
    def __init__(self):
        self.validator = SchemaValidator()
        self.benchmark = SchemaBenchmark()
        self.logger = logging.getLogger(__name__)
    
    def run_validation_tests(self) -> Dict[str, List[ValidationResult]]:
        """Exécute tous les tests de validation."""
        
        results = {
            "tenant_config": [],
            "alert_config": [],
            "monitoring_config": [],
            "compliance_config": []
        }
        
        # Tests de configuration tenant
        tenant_test_cases = [
            {
                "name": "valid_enterprise_tenant",
                "data": {
                    "tenant_id": "enterprise_001",
                    "tenant_name": "Enterprise Corp",
                    "tenant_type": "enterprise",
                    "admin_email": "admin@enterprise.com",
                    "country_code": "US",
                    "compliance_levels": ["gdpr", "soc2"],
                    "sla": {
                        "uptime_percentage": 99.99,
                        "response_time_ms": 200,
                        "support_response_minutes": 30
                    }
                }
            },
            {
                "name": "invalid_missing_required_fields",
                "data": {
                    "tenant_name": "Incomplete Tenant"
                    # Missing tenant_id and admin_email
                }
            },
            {
                "name": "invalid_low_sla",
                "data": {
                    "tenant_id": "low_sla_tenant",
                    "tenant_name": "Low SLA Tenant",
                    "admin_email": "admin@lowsla.com",
                    "sla": {
                        "uptime_percentage": 95.0  # Below minimum
                    }
                }
            }
        ]
        
        for test_case in tenant_test_cases:
            result = self.validator.validate_tenant_config(test_case["data"])
            result.schema_name = f"tenant_config_{test_case['name']}"
            results["tenant_config"].append(result)
        
        # Tests de configuration d'alerte
        alert_test_cases = [
            {
                "name": "valid_critical_alert",
                "data": {
                    "tenant_id": "test_tenant",
                    "name": "cpu_critical",
                    "title": "Critical CPU Usage",
                    "severity": "critical",
                    "category": "performance",
                    "conditions": [{
                        "metric_name": "cpu_usage_percent",
                        "operator": "gt",
                        "threshold": 90.0,
                        "duration_minutes": 5
                    }],
                    "notification_channels": ["email", "slack"],
                    "recipients": ["ops@company.com"],
                    "escalation_policy": {
                        "enabled": True,
                        "levels": [
                            {"level": 1, "delay_minutes": 5, "recipients": ["l1@company.com"]},
                            {"level": 2, "delay_minutes": 15, "recipients": ["l2@company.com"]}
                        ]
                    }
                }
            },
            {
                "name": "invalid_too_many_conditions",
                "data": {
                    "tenant_id": "test_tenant",
                    "name": "complex_alert",
                    "title": "Complex Alert",
                    "severity": "medium",
                    "category": "performance",
                    "conditions": [{"metric_name": f"metric_{i}", "operator": "gt", "threshold": 50.0} 
                                 for i in range(15)],  # Too many conditions
                    "notification_channels": ["email"]
                }
            }
        ]
        
        for test_case in alert_test_cases:
            result = self.validator.validate_alert_config(test_case["data"])
            result.schema_name = f"alert_config_{test_case['name']}"
            results["alert_config"].append(result)
        
        return results
    
    async def run_performance_tests(self) -> Dict[str, List[BenchmarkResult]]:
        """Exécute tous les tests de performance."""
        
        results = {
            "schema_creation": [],
            "schema_validation": [],
            "factory_operations": []
        }
        
        # Test data
        tenant_data = {
            "tenant_id": "perf_test_tenant",
            "tenant_name": "Performance Test Tenant",
            "tenant_type": "professional",
            "admin_email": "admin@perftest.com",
            "country_code": "US"
        }
        
        alert_data = {
            "tenant_id": "perf_test_tenant",
            "name": "perf_alert",
            "title": "Performance Alert",
            "severity": "medium",
            "category": "performance",
            "conditions": [{
                "metric_name": "response_time_ms",
                "operator": "gt",
                "threshold": 1000.0
            }],
            "notification_channels": ["email"]
        }
        
        # Benchmark création de schémas
        results["schema_creation"].append(
            self.benchmark.benchmark_schema_creation(TenantConfigSchema, tenant_data)
        )
        results["schema_creation"].append(
            self.benchmark.benchmark_schema_creation(AlertSchema, alert_data)
        )
        
        # Benchmark validation
        results["schema_validation"].append(
            self.benchmark.benchmark_schema_validation(
                self.validator, self.validator.validate_tenant_config, tenant_data
            )
        )
        results["schema_validation"].append(
            self.benchmark.benchmark_schema_validation(
                self.validator, self.validator.validate_alert_config, alert_data
            )
        )
        
        # Benchmark factory
        factory_results = await self.benchmark.benchmark_factory_creation(iterations=50)
        results["factory_operations"].extend(factory_results)
        
        return results
    
    def generate_test_report(self, validation_results: Dict[str, List[ValidationResult]],
                           performance_results: Dict[str, List[BenchmarkResult]]) -> str:
        """Génère un rapport de test complet."""
        
        report = []
        report.append("# Schema Validation & Performance Test Report")
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("")
        
        # Section validation
        report.append("## Validation Test Results")
        report.append("")
        
        for schema_type, results in validation_results.items():
            report.append(f"### {schema_type.title().replace('_', ' ')}")
            
            total_tests = len(results)
            passed_tests = len([r for r in results if r.is_valid])
            failed_tests = total_tests - passed_tests
            
            report.append(f"- Total tests: {total_tests}")
            report.append(f"- Passed: {passed_tests}")
            report.append(f"- Failed: {failed_tests}")
            report.append(f"- Success rate: {(passed_tests/total_tests*100):.1f}%")
            report.append("")
            
            for result in results:
                status = "✅ PASS" if result.is_valid else "❌ FAIL"
                report.append(f"**{result.schema_name}**: {status} ({result.validation_time_ms:.2f}ms)")
                
                if result.errors:
                    for error in result.errors:
                        report.append(f"  - ❌ {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        report.append(f"  - ⚠️  {warning}")
                
                report.append("")
        
        # Section performance
        report.append("## Performance Test Results")
        report.append("")
        
        for operation_type, results in performance_results.items():
            report.append(f"### {operation_type.title().replace('_', ' ')}")
            report.append("")
            
            for result in results:
                report.append(f"**{result.operation_name}**:")
                report.append(f"- Iterations: {result.iterations}")
                report.append(f"- Average time: {result.avg_time_ms:.2f}ms")
                report.append(f"- Min/Max time: {result.min_time_ms:.2f}ms / {result.max_time_ms:.2f}ms")
                report.append(f"- Standard deviation: {result.std_dev_ms:.2f}ms")
                report.append(f"- Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
                report.append("")
        
        return "\n".join(report)


# Fonctions utilitaires pour les tests
def create_test_data_sets() -> Dict[str, Dict[str, Any]]:
    """Crée des jeux de données de test standard."""
    
    return {
        "minimal_tenant": {
            "tenant_id": "minimal_test",
            "tenant_name": "Minimal Test Tenant",
            "admin_email": "admin@minimal.com"
        },
        "enterprise_tenant": {
            "tenant_id": "enterprise_test",
            "tenant_name": "Enterprise Test Corporation",
            "tenant_type": "enterprise",
            "admin_email": "admin@enterprise-test.com",
            "country_code": "US",
            "compliance_levels": ["gdpr", "soc2", "iso27001"],
            "features": {
                "advanced_analytics": True,
                "custom_alerts": True,
                "priority_support": True,
                "sso_integration": True,
                "api_rate_limit": 10000
            },
            "sla": {
                "uptime_percentage": 99.99,
                "response_time_ms": 100,
                "support_response_minutes": 15
            }
        },
        "complex_alert": {
            "tenant_id": "test_tenant",
            "name": "complex_performance_alert",
            "title": "Complex Performance Alert",
            "description": "Multi-condition performance alert with ML predictions",
            "severity": "high",
            "category": "performance",
            "conditions": [
                {
                    "metric_name": "cpu_usage_percent",
                    "operator": "gt",
                    "threshold": 80.0,
                    "duration_minutes": 5
                },
                {
                    "metric_name": "memory_usage_percent",
                    "operator": "gt",
                    "threshold": 85.0,
                    "duration_minutes": 3
                },
                {
                    "metric_name": "response_time_ms",
                    "operator": "gt",
                    "threshold": 2000.0,
                    "duration_minutes": 2
                }
            ],
            "notification_channels": ["email", "slack", "webhook"],
            "recipients": ["ops@company.com", "engineering@company.com"],
            "escalation_policy": {
                "enabled": True,
                "levels": [
                    {
                        "level": 1,
                        "delay_minutes": 5,
                        "recipients": ["l1-ops@company.com"],
                        "channels": ["email", "slack"]
                    },
                    {
                        "level": 2,
                        "delay_minutes": 15,
                        "recipients": ["l2-engineering@company.com"],
                        "channels": ["email", "phone"]
                    },
                    {
                        "level": 3,
                        "delay_minutes": 30,
                        "recipients": ["management@company.com"],
                        "channels": ["email", "phone", "sms"]
                    }
                ]
            },
            "ml_prediction": {
                "enabled": True,
                "model_name": "performance_anomaly_detector",
                "confidence_threshold": 0.8,
                "prediction_window_minutes": 15
            }
        }
    }


async def main():
    """Fonction principale pour exécuter tous les tests."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Créer la suite de tests
    test_suite = SchemaTestSuite()
    
    print("🧪 Starting schema validation and performance tests...")
    
    # Exécuter les tests de validation
    print("📋 Running validation tests...")
    validation_results = test_suite.run_validation_tests()
    
    # Exécuter les tests de performance
    print("🚀 Running performance tests...")
    performance_results = await test_suite.run_performance_tests()
    
    # Générer le rapport
    print("📊 Generating test report...")
    report = test_suite.generate_test_report(validation_results, performance_results)
    
    # Sauvegarder le rapport
    report_path = Path("/tmp/schema_test_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Test report generated: {report_path}")
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    # Résumé des validations
    total_validation_tests = sum(len(results) for results in validation_results.values())
    total_passed = sum(len([r for r in results if r.is_valid]) for results in validation_results.values())
    
    print(f"Validation Tests: {total_passed}/{total_validation_tests} passed ({total_passed/total_validation_tests*100:.1f}%)")
    
    # Résumé des performances
    print("\nPerformance Highlights:")
    for operation_type, results in performance_results.items():
        for result in results:
            print(f"- {result.operation_name}: {result.avg_time_ms:.2f}ms avg, {result.throughput_ops_per_sec:.2f} ops/sec")


if __name__ == "__main__":
    asyncio.run(main())
