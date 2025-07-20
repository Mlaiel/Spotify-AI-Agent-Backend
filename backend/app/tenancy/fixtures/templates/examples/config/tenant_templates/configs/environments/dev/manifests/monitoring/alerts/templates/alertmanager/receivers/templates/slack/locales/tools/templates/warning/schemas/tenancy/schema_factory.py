#!/usr/bin/env python3
"""
Advanced Schema Factory and Builder
===================================

Factory pattern pour la création automatisée de schémas tenancy
avec validation, optimisation et intégration machine learning.
"""

import json
import yaml
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import des schémas
from .tenant_config_schema import (
    TenantConfigSchema, TenantType, TenantStatus, 
    TenantFeatures, TenantSLA, TenantSecurityConfig
)
from .alert_schema import (
    AlertSchema, TenantAlertSchema, AlertSeverity, 
    AlertCategory, AlertCondition, AlertAction
)
from .warning_schema import (
    WarningSchema, TenantWarningSchema, WarningSeverity,
    WarningType, WarningMetric, WarningPrediction
)
from .notification_schema import (
    NotificationSchema, NotificationChannel, NotificationPriority,
    NotificationRecipient, NotificationTemplate
)
from .monitoring_schema import (
    MonitoringConfigSchema, MonitoringMetric, MonitoringDashboard,
    MonitoringTarget, AlertRule
)
from .compliance_schema import (
    ComplianceSchema, ComplianceStandard, ComplianceControl,
    DataRetentionPolicy, AuditTrail
)
from .performance_schema import (
    PerformanceMetricsSchema, PerformanceBaseline,
    AnomalyDetection, PerformanceTrend
)


@dataclass
class SchemaBuilderConfig:
    """Configuration pour le builder de schémas."""
    tenant_type: TenantType
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    features_override: Dict[str, Any] = field(default_factory=dict)
    environment: str = "production"
    region: str = "us-east-1"
    auto_optimize: bool = True
    ml_predictions_enabled: bool = True
    custom_templates: Dict[str, Any] = field(default_factory=dict)


class SchemaOptimizer:
    """Optimiseur de schémas avec ML et heuristiques."""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
        self.ml_models = self._initialize_ml_models()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Charge les règles d'optimisation."""
        return {
            "tenant_config": {
                "enterprise": {
                    "features": {
                        "advanced_analytics": True,
                        "custom_alerts": True,
                        "priority_support": True,
                        "sso_integration": True
                    },
                    "sla": {
                        "uptime_percentage": 99.99,
                        "response_time_ms": 200,
                        "support_response_minutes": 30
                    }
                },
                "professional": {
                    "features": {
                        "advanced_analytics": True,
                        "custom_alerts": True,
                        "max_users": 500
                    },
                    "sla": {
                        "uptime_percentage": 99.9,
                        "response_time_ms": 500
                    }
                }
            },
            "alerts": {
                "critical_auto_escalation": True,
                "min_escalation_levels": 2,
                "max_notification_channels": 5
            }
        }
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialise les modèles ML pour l'optimisation."""
        return {
            "threshold_optimizer": None,  # Modèle pour optimiser les seuils
            "escalation_predictor": None,  # Modèle pour prédire les escalations
            "performance_predictor": None  # Modèle pour prédire les performances
        }
    
    def optimize_tenant_config(self, config: TenantConfigSchema) -> TenantConfigSchema:
        """Optimise la configuration d'un tenant."""
        optimized_data = config.dict()
        
        # Optimisations basées sur le type de tenant
        tenant_type = config.tenant_type
        if tenant_type in self.optimization_rules["tenant_config"]:
            rules = self.optimization_rules["tenant_config"][tenant_type]
            
            # Optimiser les features
            if "features" in rules:
                for feature, value in rules["features"].items():
                    setattr(optimized_data["features"], feature, value)
            
            # Optimiser les SLA
            if "sla" in rules:
                for sla_key, value in rules["sla"].items():
                    setattr(optimized_data["sla"], sla_key, value)
        
        return TenantConfigSchema(**optimized_data)
    
    def optimize_alert_thresholds(self, alerts: List[AlertSchema], 
                                performance_data: Dict[str, Any]) -> List[AlertSchema]:
        """Optimise les seuils d'alerte basés sur les données de performance."""
        optimized_alerts = []
        
        for alert in alerts:
            optimized_alert_data = alert.dict()
            
            # Optimiser les conditions basées sur les données historiques
            for i, condition in enumerate(optimized_alert_data["conditions"]):
                metric_name = condition["metric_name"]
                
                if metric_name in performance_data:
                    historical_data = performance_data[metric_name]
                    
                    # Calculer le seuil optimal basé sur les percentiles
                    if condition["operator"] in ["gt", "gte"]:
                        # Pour les métriques "plus grand que", utiliser le 95e percentile
                        optimal_threshold = historical_data.get("p95", condition["threshold"])
                        condition["threshold"] = optimal_threshold * 1.1  # 10% de marge
                    elif condition["operator"] in ["lt", "lte"]:
                        # Pour les métriques "plus petit que", utiliser le 5e percentile
                        optimal_threshold = historical_data.get("p5", condition["threshold"])
                        condition["threshold"] = optimal_threshold * 0.9  # 10% de marge
            
            optimized_alerts.append(AlertSchema(**optimized_alert_data))
        
        return optimized_alerts


class SchemaFactory:
    """Factory pour créer des schémas optimisés."""
    
    def __init__(self, config: SchemaBuilderConfig):
        self.config = config
        self.optimizer = SchemaOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def create_tenant_config(self, tenant_id: str, tenant_name: str, 
                           admin_email: str, **kwargs) -> TenantConfigSchema:
        """Crée une configuration tenant optimisée."""
        
        # Configuration de base
        base_config = {
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "tenant_type": self.config.tenant_type,
            "admin_email": admin_email,
            "country_code": kwargs.get("country_code", "US"),
            "compliance_levels": self.config.compliance_standards,
            "environment": self.config.environment
        }
        
        # Appliquer les overrides de features
        if self.config.features_override:
            base_config["features"] = self.config.features_override
        
        # Créer le schéma
        tenant_config = TenantConfigSchema(**base_config)
        
        # Optimiser si activé
        if self.config.auto_optimize:
            tenant_config = self.optimizer.optimize_tenant_config(tenant_config)
        
        return tenant_config
    
    def create_monitoring_config(self, tenant_id: str, 
                               services: List[str] = None) -> MonitoringConfigSchema:
        """Crée une configuration de monitoring complète."""
        
        services = services or ["api", "database", "cache", "queue"]
        
        # Métriques par défaut
        default_metrics = []
        for service in services:
            default_metrics.extend([
                MonitoringMetric(
                    metric_id=f"{service}_response_time",
                    name=f"{service}_response_time_ms",
                    display_name=f"{service.title()} Response Time",
                    description=f"Response time for {service} service",
                    metric_type="histogram",
                    source="application",
                    query=f"avg({service}_response_time_ms)",
                    unit="milliseconds",
                    warning_threshold=500.0,
                    critical_threshold=2000.0
                ),
                MonitoringMetric(
                    metric_id=f"{service}_error_rate",
                    name=f"{service}_error_rate_percent",
                    display_name=f"{service.title()} Error Rate",
                    description=f"Error rate for {service} service",
                    metric_type="gauge",
                    source="application",
                    query=f"rate({service}_errors_total[5m]) * 100",
                    unit="percentage",
                    warning_threshold=5.0,
                    critical_threshold=10.0
                )
            ])
        
        # Configuration de monitoring
        monitoring_config = MonitoringConfigSchema(
            tenant_id=tenant_id,
            name=f"Monitoring Config for {tenant_id}",
            description="Auto-generated monitoring configuration",
            metrics=default_metrics,
            collection_interval_seconds=30,
            retention_days=90,
            anomaly_detection_enabled=True,
            trend_analysis_enabled=True
        )
        
        return monitoring_config
    
    def create_compliance_config(self, tenant_id: str) -> ComplianceSchema:
        """Crée une configuration de compliance."""
        
        # Contrôles par standard
        controls = []
        
        for standard in self.config.compliance_standards:
            if standard == ComplianceStandard.GDPR:
                controls.extend(self._create_gdpr_controls())
            elif standard == ComplianceStandard.SOC2:
                controls.extend(self._create_soc2_controls())
            elif standard == ComplianceStandard.HIPAA:
                controls.extend(self._create_hipaa_controls())
            elif standard == ComplianceStandard.ISO27001:
                controls.extend(self._create_iso27001_controls())
        
        # Politiques de rétention
        retention_policies = self._create_retention_policies()
        
        compliance_config = ComplianceSchema(
            tenant_id=tenant_id,
            name=f"Compliance Program for {tenant_id}",
            applicable_standards=self.config.compliance_standards,
            compliance_framework="NIST CSF",
            controls=controls,
            data_retention_policies=retention_policies,
            compliance_officer=f"compliance-{tenant_id}@company.com",
            monitoring_enabled=True,
            automated_reporting=True
        )
        
        return compliance_config
    
    def _create_gdpr_controls(self) -> List[ComplianceControl]:
        """Crée les contrôles GDPR."""
        return [
            ComplianceControl(
                control_id="GDPR-001",
                name="Data Processing Lawfulness",
                description="Ensure all data processing has a lawful basis",
                control_type="preventive",
                category="data_protection",
                standards=[ComplianceStandard.GDPR],
                requirements=["Lawful basis identification", "Consent management"],
                status="compliant",
                risk_level="high",
                implementation_status="implemented",
                responsible_party="privacy-team@company.com"
            ),
            ComplianceControl(
                control_id="GDPR-002",
                name="Right to Erasure",
                description="Enable data subjects to request deletion of personal data",
                control_type="corrective",
                category="data_subject_rights",
                standards=[ComplianceStandard.GDPR],
                requirements=["Deletion mechanism", "Verification process"],
                status="compliant",
                risk_level="medium",
                implementation_status="implemented",
                responsible_party="engineering-team@company.com"
            )
        ]
    
    def _create_soc2_controls(self) -> List[ComplianceControl]:
        """Crée les contrôles SOC2."""
        return [
            ComplianceControl(
                control_id="SOC2-CC6.1",
                name="Logical Access Controls",
                description="Restrict logical access to information assets",
                control_type="preventive",
                category="access_control",
                standards=[ComplianceStandard.SOC2],
                requirements=["User authentication", "Authorization controls"],
                status="compliant",
                risk_level="high",
                implementation_status="implemented",
                responsible_party="security-team@company.com"
            )
        ]
    
    def _create_hipaa_controls(self) -> List[ComplianceControl]:
        """Crée les contrôles HIPAA."""
        return [
            ComplianceControl(
                control_id="HIPAA-164.308",
                name="Administrative Safeguards",
                description="Implement administrative safeguards for PHI",
                control_type="directive",
                category="administrative",
                standards=[ComplianceStandard.HIPAA],
                requirements=["Security officer", "Workforce training"],
                status="compliant",
                risk_level="high",
                implementation_status="implemented",
                responsible_party="compliance-team@company.com"
            )
        ]
    
    def _create_iso27001_controls(self) -> List[ComplianceControl]:
        """Crée les contrôles ISO27001."""
        return [
            ComplianceControl(
                control_id="ISO-A.9.1.1",
                name="Access Control Policy",
                description="Establish, document and review access control policy",
                control_type="directive",
                category="access_control",
                standards=[ComplianceStandard.ISO27001],
                requirements=["Policy documentation", "Regular review"],
                status="compliant",
                risk_level="medium",
                implementation_status="implemented",
                responsible_party="security-team@company.com"
            )
        ]
    
    def _create_retention_policies(self) -> List[DataRetentionPolicy]:
        """Crée les politiques de rétention."""
        return [
            DataRetentionPolicy(
                policy_id="user_data_retention",
                name="User Data Retention Policy",
                data_types=["user_profiles", "preferences", "activity_logs"],
                data_classification="confidential",
                contains_pii=True,
                retention_period_days=1095,  # 3 ans
                auto_deletion_enabled=True,
                deletion_method="anonymization",
                applicable_standards=self.config.compliance_standards,
                created_by="system",
                effective_date=datetime.now(timezone.utc),
                review_date=datetime.now(timezone.utc) + timedelta(days=365)
            )
        ]


class SchemaBuilder:
    """Builder principal pour créer des configurations complètes."""
    
    def __init__(self, config: SchemaBuilderConfig):
        self.config = config
        self.factory = SchemaFactory(config)
        self.logger = logging.getLogger(__name__)
    
    async def build_complete_tenant(self, tenant_id: str, tenant_name: str, 
                                  admin_email: str, **kwargs) -> Dict[str, Any]:
        """Construit une configuration complète de tenant."""
        
        self.logger.info(f"Building complete tenant configuration for {tenant_id}")
        
        # Créer les configurations en parallèle
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Configuration tenant
            tenant_config_future = executor.submit(
                self.factory.create_tenant_config,
                tenant_id, tenant_name, admin_email, **kwargs
            )
            
            # Configuration monitoring
            monitoring_config_future = executor.submit(
                self.factory.create_monitoring_config,
                tenant_id, kwargs.get("services", ["api", "database"])
            )
            
            # Configuration compliance
            compliance_config_future = executor.submit(
                self.factory.create_compliance_config,
                tenant_id
            )
            
            # Attendre les résultats
            tenant_config = tenant_config_future.result()
            monitoring_config = monitoring_config_future.result()
            compliance_config = compliance_config_future.result()
        
        # Créer des alertes par défaut
        default_alerts = self._create_default_alerts(tenant_id, tenant_config.tenant_type)
        
        # Créer des notifications par défaut
        default_notifications = self._create_default_notifications(tenant_id, admin_email)
        
        # Configuration de performance
        performance_config = PerformanceMetricsSchema(
            tenant_id=tenant_id,
            name=f"Performance Metrics for {tenant_id}",
            collection_interval_seconds=60,
            retention_days=90,
            anomaly_detection_enabled=True,
            trend_analysis_enabled=True,
            auto_optimization_enabled=self.config.auto_optimize
        )
        
        result = {
            "tenant_config": tenant_config,
            "monitoring_config": monitoring_config,
            "compliance_config": compliance_config,
            "alerts": default_alerts,
            "notifications": default_notifications,
            "performance_config": performance_config,
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "builder_version": "1.0.0",
                "config_version": self.config.tenant_type.value,
                "auto_optimized": self.config.auto_optimize
            }
        }
        
        self.logger.info(f"Complete tenant configuration built for {tenant_id}")
        return result
    
    def _create_default_alerts(self, tenant_id: str, tenant_type: TenantType) -> List[AlertSchema]:
        """Crée les alertes par défaut selon le type de tenant."""
        
        alerts = []
        
        # Alertes communes
        common_alerts = [
            {
                "name": "high_cpu_usage",
                "title": "High CPU Usage",
                "description": "CPU usage exceeded threshold",
                "severity": "high",
                "category": "performance",
                "conditions": [{
                    "metric_name": "cpu_usage_percent",
                    "operator": "gt",
                    "threshold": 80.0,
                    "duration_minutes": 5
                }]
            },
            {
                "name": "high_memory_usage",
                "title": "High Memory Usage",
                "description": "Memory usage exceeded threshold",
                "severity": "high",
                "category": "performance",
                "conditions": [{
                    "metric_name": "memory_usage_percent",
                    "operator": "gt",
                    "threshold": 85.0,
                    "duration_minutes": 5
                }]
            },
            {
                "name": "api_error_rate",
                "title": "High API Error Rate",
                "description": "API error rate exceeded threshold",
                "severity": "critical",
                "category": "application",
                "conditions": [{
                    "metric_name": "api_error_rate_percent",
                    "operator": "gt",
                    "threshold": 5.0,
                    "duration_minutes": 2
                }]
            }
        ]
        
        # Alertes spécifiques par type de tenant
        if tenant_type == TenantType.ENTERPRISE:
            common_alerts.extend([
                {
                    "name": "sla_breach_warning",
                    "title": "SLA Breach Warning",
                    "description": "SLA metrics approaching breach threshold",
                    "severity": "critical",
                    "category": "business",
                    "conditions": [{
                        "metric_name": "sla_achievement_percent",
                        "operator": "lt",
                        "threshold": 99.5,
                        "duration_minutes": 1
                    }]
                }
            ])
        
        # Convertir en objets AlertSchema
        for alert_data in common_alerts:
            alert_data.update({
                "tenant_id": tenant_id,
                "notification_channels": ["email", "slack"],
                "recipients": [f"ops-{tenant_id}@company.com"]
            })
            
            alerts.append(AlertSchema(**alert_data))
        
        return alerts
    
    def _create_default_notifications(self, tenant_id: str, admin_email: str) -> List[NotificationSchema]:
        """Crée les notifications par défaut."""
        
        notifications = [
            NotificationSchema(
                tenant_id=tenant_id,
                notification_type="system",
                priority="normal",
                title="Tenant Configuration Complete",
                message=f"Tenant {tenant_id} has been successfully configured and is ready for use.",
                preferred_channels=["email"],
                recipients=[
                    NotificationRecipient(
                        recipient_id="admin",
                        recipient_type="user",
                        contact_info={"email": admin_email},
                        timezone="UTC"
                    )
                ]
            )
        ]
        
        return notifications
    
    def export_configuration(self, config_data: Dict[str, Any], 
                           output_path: Path, format: str = "json") -> None:
        """Exporte la configuration vers un fichier."""
        
        # Convertir les objets Pydantic en dictionnaires
        exportable_data = {}
        for key, value in config_data.items():
            if hasattr(value, 'dict'):
                exportable_data[key] = value.dict()
            elif isinstance(value, list) and value and hasattr(value[0], 'dict'):
                exportable_data[key] = [item.dict() for item in value]
            else:
                exportable_data[key] = value
        
        # Exporter selon le format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(exportable_data, f, indent=2, default=str)
        elif format.lower() in ["yaml", "yml"]:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(exportable_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration exported to {output_path}")


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation du SchemaBuilder."""
    
    # Configuration du builder
    config = SchemaBuilderConfig(
        tenant_type=TenantType.ENTERPRISE,
        compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOC2],
        environment="production",
        auto_optimize=True,
        ml_predictions_enabled=True
    )
    
    # Créer le builder
    builder = SchemaBuilder(config)
    
    # Construire une configuration complète
    tenant_config = await builder.build_complete_tenant(
        tenant_id="enterprise_demo_001",
        tenant_name="Demo Enterprise Corporation",
        admin_email="admin@demo-enterprise.com",
        country_code="US",
        services=["api", "database", "cache", "queue", "ml_service"]
    )
    
    # Exporter la configuration
    output_path = Path("/tmp/tenant_configs/enterprise_demo_001.json")
    builder.export_configuration(tenant_config, output_path, "json")
    
    print(f"Configuration complète créée et exportée vers {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
