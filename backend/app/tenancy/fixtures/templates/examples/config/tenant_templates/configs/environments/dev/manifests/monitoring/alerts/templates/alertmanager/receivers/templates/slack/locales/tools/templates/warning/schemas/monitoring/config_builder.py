#!/usr/bin/env python3
"""
Advanced Monitoring Configuration Builder
========================================

Script pour créer des configurations avancées spécifiques aux besoins
du projet Spotify AI Agent avec exemples et templates.

Usage:
    python config_builder.py --build-spotify-config
    python config_builder.py --create-examples
    python config_builder.py --validate-all
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from metric_schemas import *
from alert_schemas import *
from dashboard_schemas import *
from tenant_monitoring import *
from compliance_monitoring import *
from ml_monitoring import *
from security_monitoring import *
from performance_monitoring import *

logger = logging.getLogger(__name__)


class SpotifyMonitoringConfigBuilder:
    """Builder pour configuration monitoring Spotify AI Agent"""
    
    def __init__(self):
        self.config = {}
        
    def build_comprehensive_spotify_config(self) -> Dict[str, Any]:
        """Construire configuration complète pour Spotify AI Agent"""
        logger.info("Building comprehensive Spotify monitoring configuration...")
        
        # Configuration des métriques avancées
        metrics_config = self._build_spotify_metrics()
        
        # Configuration des alertes intelligentes
        alerts_config = self._build_spotify_alerts()
        
        # Configuration des dashboards exécutifs
        dashboards_config = self._build_spotify_dashboards()
        
        # Configuration multi-tenant
        tenant_config = self._build_spotify_tenants()
        
        # Configuration conformité
        compliance_config = self._build_spotify_compliance()
        
        # Configuration ML/IA
        ml_config = self._build_spotify_ml_monitoring()
        
        # Configuration sécurité
        security_config = self._build_spotify_security()
        
        # Configuration performance
        performance_config = self._build_spotify_performance()
        
        return {
            'project': 'spotify-ai-agent',
            'version': '3.0.0',
            'generated_at': datetime.utcnow().isoformat(),
            'environment': 'production',
            'configuration': {
                'metrics': metrics_config,
                'alerts': alerts_config,
                'dashboards': dashboards_config,
                'tenant_monitoring': tenant_config,
                'compliance': compliance_config,
                'ml_monitoring': ml_config,
                'security': security_config,
                'performance': performance_config
            }
        }
    
    def _build_spotify_metrics(self) -> Dict[str, Any]:
        """Construire métriques spécifiques Spotify"""
        
        # Métriques business critiques
        spotify_business_metrics = [
            MetricSchema(
                name="spotify_user_engagement_rate",
                display_name="User Engagement Rate",
                description="Taux d'engagement utilisateur avec détection anomalies ML",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.BUSINESS,
                severity=MetricSeverity.CRITICAL,
                dimensions=[
                    MetricDimension(name="user_tier", value="premium", cardinality=3),
                    MetricDimension(name="region", value="global", cardinality=50),
                    MetricDimension(name="platform", value="mobile", cardinality=5)
                ],
                unit="%",
                thresholds=MetricThreshold(warning=75.0, critical=60.0, operator="lt"),
                ml_config=MLAnomalyConfig(
                    enabled=True,
                    model_type="lstm_autoencoder",
                    sensitivity=0.85,
                    features=["daily_listening_hours", "tracks_played", "skips_ratio"]
                ),
                is_sli=True,
                slo_target=85.0,
                business_impact="Direct impact on user retention and revenue",
                cost_center="growth_marketing"
            ),
            
            MetricSchema(
                name="spotify_recommendation_click_through_rate",
                display_name="Recommendation CTR",
                description="Taux de clic sur recommandations avec analyse ML",
                metric_type=MetricType.RATE,
                category=MetricCategory.ML_AI,
                severity=MetricSeverity.HIGH,
                unit="%",
                ml_config=MLAnomalyConfig(enabled=True, model_type="isolation_forest"),
                is_sli=True,
                slo_target=12.5
            ),
            
            MetricSchema(
                name="spotify_revenue_per_user_hour",
                display_name="Revenue per User Hour",
                description="Revenus par heure utilisateur avec prédiction ML",
                metric_type=MetricType.BUSINESS_KPI,
                category=MetricCategory.FINANCIAL,
                severity=MetricSeverity.CRITICAL,
                unit="USD",
                ml_config=MLAnomalyConfig(
                    enabled=True,
                    model_type="prophet",
                    prediction_window="24h"
                ),
                business_impact="Direct revenue impact measurement"
            )
        ]
        
        # Métriques techniques avancées
        spotify_technical_metrics = [
            MetricSchema(
                name="spotify_api_ml_inference_latency",
                display_name="ML Inference Latency",
                description="Latence inférence ML pour recommandations",
                metric_type=MetricType.LATENCY,
                category=MetricCategory.ML_AI,
                severity=MetricSeverity.HIGH,
                unit="ms",
                thresholds=MetricThreshold(warning=200.0, critical=500.0, operator="gt"),
                dimensions=[
                    MetricDimension(name="model_version", value="v3.2.1", cardinality=10),
                    MetricDimension(name="inference_type", value="recommendation", cardinality=5)
                ]
            ),
            
            MetricSchema(
                name="spotify_data_pipeline_freshness",
                display_name="Data Pipeline Freshness",
                description="Fraîcheur des données pour ML avec alertes intelligentes",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.OPERATIONAL,
                severity=MetricSeverity.HIGH,
                unit="minutes",
                thresholds=MetricThreshold(warning=30.0, critical=60.0, operator="gt")
            )
        ]
        
        return {
            'business_metrics': [m.dict() for m in spotify_business_metrics],
            'technical_metrics': [m.dict() for m in spotify_technical_metrics],
            'total_metrics': len(spotify_business_metrics) + len(spotify_technical_metrics)
        }
    
    def _build_spotify_alerts(self) -> Dict[str, Any]:
        """Construire alertes intelligentes Spotify"""
        
        spotify_critical_alerts = [
            AlertRule(
                name="spotify_revenue_drop_anomaly",
                display_name="Revenue Drop Anomaly Detected",
                description="Anomalie de chute de revenus détectée par ML",
                metric_name="spotify_revenue_per_user_hour",
                query="avg(spotify_revenue_per_user_hour) < 0.8 * avg_over_time(spotify_revenue_per_user_hour[7d])",
                condition="Revenue drops below 80% of 7-day average",
                threshold=0.8,
                comparison_operator="lt",
                priority=AlertPriority.P0_CRITICAL,
                severity=MetricSeverity.CRITICAL,
                category=MetricCategory.FINANCIAL,
                for_duration="5m",
                use_ml_prediction=True,
                anomaly_detection=True,
                sla_impact=True,
                business_impact_score=10.0,
                cost_impact=100000.0,
                labels={
                    "severity": "critical",
                    "team": "growth",
                    "impact": "revenue",
                    "escalation": "immediate"
                },
                annotations={
                    "summary": "Critical revenue anomaly detected",
                    "description": "ML models detected significant revenue drop anomaly",
                    "runbook": "https://runbooks.spotify.internal/revenue-anomaly",
                    "dashboard": "https://grafana.spotify.internal/revenue-monitoring"
                },
                notification_templates=[
                    NotificationTemplate(
                        channel=NotificationChannel.SLACK,
                        template_id="revenue_critical",
                        subject_template="🚨 CRITICAL: Revenue Anomaly - {{business_impact_score}}/10",
                        body_template="""
🚨 **CRITICAL REVENUE ANOMALY DETECTED**

**Service**: Spotify AI Agent
**Metric**: Revenue per User Hour
**Current Value**: {{current_value}} USD
**Expected Range**: {{expected_min}} - {{expected_max}} USD
**Anomaly Score**: {{anomaly_score}}

**Business Impact**: {{business_impact_score}}/10
**Estimated Cost Impact**: ${{cost_impact}}

**Immediate Actions Required**:
• Check payment processing systems
• Verify user engagement metrics  
• Review recent A/B tests
• Escalate to Growth team lead

**Runbook**: {{runbook_url}}
**Dashboard**: {{dashboard_url}}
                        """,
                        channel_config={
                            "channel": "#critical-alerts",
                            "username": "RevenueBot",
                            "icon_emoji": ":money_with_wings:"
                        }
                    )
                ]
            ),
            
            AlertRule(
                name="spotify_ml_model_accuracy_degradation",
                display_name="ML Model Accuracy Degradation",
                description="Dégradation critique de précision modèles ML",
                metric_name="spotify_recommendation_accuracy",
                query="avg(spotify_recommendation_accuracy) < 0.85",
                threshold=0.85,
                comparison_operator="lt",
                priority=AlertPriority.P1_HIGH,
                severity=MetricSeverity.HIGH,
                category=MetricCategory.ML_AI,
                for_duration="10m",
                use_ml_prediction=True,
                trend_analysis=True,
                business_impact_score=8.5,
                labels={
                    "team": "ml_engineering",
                    "model_type": "recommendation",
                    "impact": "user_experience"
                },
                auto_remediation=AutoRemediationRule(
                    name="trigger_model_retrain",
                    condition="accuracy < 0.80",
                    action=RemediationAction.CUSTOM_SCRIPT,
                    action_params={"script": "trigger_emergency_retrain.py"},
                    requires_approval=False,
                    max_attempts=1
                )
            )
        ]
        
        return {
            'critical_alerts': [a.dict() for a in spotify_critical_alerts],
            'total_rules': len(spotify_critical_alerts)
        }
    
    def _build_spotify_dashboards(self) -> Dict[str, Any]:
        """Construire dashboards exécutifs Spotify"""
        
        # Dashboard C-Level Executive
        executive_dashboard = Dashboard(
            id="spotify_c_level_executive",
            title="Spotify AI Agent - C-Level Executive Overview",
            description="Vue stratégique pour dirigeants avec KPIs critiques et prédictions IA",
            dashboard_type=DashboardType.EXECUTIVE,
            category=MetricCategory.BUSINESS,
            default_time_range=TimeRange.LAST_24_HOURS,
            ml_insights_enabled=True,
            predictive_analytics=True
        )
        
        # Widgets exécutifs
        revenue_kpi = Widget(
            id="revenue_impact_kpi",
            title="Revenue Impact (24H)",
            visualization_type=VisualizationType.SINGLE_STAT,
            category=MetricCategory.FINANCIAL,
            query="sum(increase(spotify_revenue_total[24h]))",
            position={"x": 0, "y": 0},
            size=WidgetSize.LARGE,
            format_type="currency",
            ml_config=MLVisualizationConfig(
                show_predictions=True,
                show_confidence_bands=True,
                prediction_color="#2196f3"
            )
        )
        
        user_growth = Widget(
            id="user_growth_trend",
            title="User Growth Trend with ML Predictions",
            visualization_type=VisualizationType.LINE_CHART,
            category=MetricCategory.BUSINESS,
            query="increase(spotify_active_users_total[1h])",
            position={"x": 4, "y": 0},
            size=WidgetSize.XLARGE,
            ml_config=MLVisualizationConfig(
                show_anomalies=True,
                show_predictions=True,
                show_confidence_bands=True
            )
        )
        
        executive_dashboard.add_widget(revenue_kpi)
        executive_dashboard.add_widget(user_growth)
        
        # Dashboard Real-Time Operations
        ops_dashboard = Dashboard(
            id="spotify_realtime_operations",
            title="Spotify AI Agent - Real-Time Operations",
            description="Monitoring opérationnel temps réel avec IA prédictive",
            dashboard_type=DashboardType.OPERATIONAL,
            category=MetricCategory.APPLICATION,
            default_refresh_interval=RefreshInterval.FIVE_SECONDS,
            anomaly_detection_enabled=True
        )
        
        return {
            'executive_dashboard': executive_dashboard.dict(),
            'operations_dashboard': ops_dashboard.dict(),
            'total_dashboards': 2
        }
    
    def _build_spotify_tenants(self) -> Dict[str, Any]:
        """Construire configuration multi-tenant Spotify"""
        
        # Tenant Premium Enterprise
        premium_enterprise = TenantConfig(
            tenant_id="spotify-premium-enterprise",
            tenant_name="Spotify Premium Enterprise",
            display_name="Spotify Premium Enterprise Customers",
            tier=TenantTier.ENTERPRISE,
            status=TenantStatus.ACTIVE,
            isolation_level=IsolationLevel.STRICT,
            quotas=TenantQuota(
                max_metrics=100000,
                max_cardinality=50000000,
                ingestion_rate=100000,
                max_storage_gb=1000.0,
                max_cpu_cores=32.0,
                max_memory_gb=64.0
            ),
            sla=TenantSLA(
                availability_target=99.99,
                response_time_p95=100.0,
                response_time_p99=250.0,
                max_error_rate=0.001,
                rto="30m",
                rpo="5m"
            ),
            security=TenantSecurityConfig(
                encryption_at_rest=True,
                encryption_in_transit=True,
                mfa_required=True,
                compliance_frameworks=["GDPR", "SOC2", "HIPAA", "ISO27001"]
            ),
            ml_config=TenantMLConfig(
                enable_anomaly_detection=True,
                enable_predictive_analytics=True,
                enable_auto_scaling=True,
                differential_privacy=True
            ),
            namespace="tenant-premium-enterprise",
            database_schema="tenant_premium_enterprise"
        )
        
        # Tenant Standard
        standard_tenant = TenantConfig(
            tenant_id="spotify-standard",
            tenant_name="Spotify Standard Tier",
            tier=TenantTier.PROFESSIONAL,
            isolation_level=IsolationLevel.STANDARD,
            namespace="tenant-standard",
            database_schema="tenant_standard"
        )
        
        return {
            'premium_enterprise': premium_enterprise.dict(),
            'standard_tier': standard_tenant.dict(),
            'total_tenants': 2
        }
    
    def _build_spotify_compliance(self) -> Dict[str, Any]:
        """Construire monitoring conformité Spotify"""
        
        # Contrôles GDPR spécifiques musique
        gdpr_music_controls = [
            ComplianceControl(
                control_id="gdpr_music_consent_management",
                framework=ComplianceFramework.GDPR,
                control_number="Art. 7",
                title="Music Preferences Consent Management",
                description="Gestion du consentement pour préférences musicales",
                objective="Garantir consentement valide pour données musicales",
                category="Consent Management",
                criticality=RiskLevel.CRITICAL,
                inherent_risk=RiskLevel.HIGH,
                residual_risk=RiskLevel.LOW,
                automated_check=True,
                check_frequency="1h",
                check_query="spotify_invalid_consent_count > 0",
                current_status=ComplianceStatus.COMPLIANT,
                last_assessment_date=datetime.utcnow(),
                next_assessment_date=datetime.utcnow() + timedelta(days=7),
                owner="Data Protection Officer",
                reviewer="Legal Team"
            )
        ]
        
        return {
            'gdpr_controls': [c.dict() for c in gdpr_music_controls],
            'active_frameworks': ["GDPR", "SOC2", "ISO27001"],
            'total_controls': len(gdpr_music_controls)
        }
    
    def _build_spotify_ml_monitoring(self) -> Dict[str, Any]:
        """Construire monitoring ML Spotify"""
        
        # Configuration modèles ML avancés
        advanced_recommendation_model = MLModelConfig(
            model_id="spotify_advanced_recommender_v4",
            model_name="Advanced Music Recommender",
            model_version="4.0.0",
            model_type=ModelType.RECOMMENDATION,
            model_stage=ModelStage.PRODUCTION,
            use_case="Advanced personalized music recommendations with context",
            business_objective="Increase user engagement and discovery",
            owner="ML Platform Team",
            team="Recommendation Systems",
            framework="TensorFlow",
            framework_version="2.13.0",
            input_features=[
                "user_listening_history_embeddings",
                "track_audio_features",
                "contextual_signals",
                "social_graph_features",
                "temporal_patterns",
                "device_context",
                "location_context"
            ],
            performance_thresholds={
                "ndcg_at_10": 0.85,
                "precision_at_5": 0.80,
                "recall_at_20": 0.75,
                "diversity_score": 0.70,
                "novelty_score": 0.65
            },
            latency_sla_ms=150.0,
            throughput_sla_rps=1000.0,
            tags=["recommendation", "deep_learning", "production", "critical"]
        )
        
        return {
            'recommendation_model': advanced_recommendation_model.dict(),
            'total_models': 1
        }
    
    def _build_spotify_security(self) -> Dict[str, Any]:
        """Construire monitoring sécurité Spotify"""
        
        # Événements sécurité spécifiques streaming musical
        music_security_events = [
            {
                "threat_type": "unauthorized_access",
                "threat_level": "high",
                "title": "Unauthorized Music Content Access",
                "description": "Tentative d'accès non autorisé à contenu musical premium",
                "mitre_tactics": ["initial_access", "credential_access"],
                "detection_rules": [
                    "Multiple failed premium content access attempts",
                    "Unusual geographic access patterns",
                    "Suspicious API usage patterns"
                ]
            }
        ]
        
        return {
            'music_specific_threats': music_security_events,
            'threat_intelligence_feeds': ["music_piracy_intel", "streaming_threats"],
            'total_threat_types': len(music_security_events)
        }
    
    def _build_spotify_performance(self) -> Dict[str, Any]:
        """Construire monitoring performance Spotify"""
        
        # SLOs critiques pour streaming musical
        music_streaming_slos = [
            SLO(
                slo_id="spotify_music_streaming_latency",
                name="Music Streaming Latency",
                description="Latence de streaming musical sous 100ms",
                service_name="music-streaming-service",
                service_tier=ServiceTier.CRITICAL,
                target_percentage=99.5,
                measurement_window="30d",
                sli_metric="streaming_latency",
                sli_query="histogram_quantile(0.95, streaming_latency_bucket)",
                good_event_query="sum(rate(streaming_latency_bucket{le='0.1'}[5m]))",
                total_event_query="sum(rate(streaming_latency_count[5m]))",
                owner="Streaming Team",
                team="Platform Engineering"
            ),
            
            SLO(
                slo_id="spotify_recommendation_accuracy",
                name="Recommendation Accuracy",
                description="Précision des recommandations musicales",
                service_name="recommendation-service",
                service_tier=ServiceTier.CRITICAL,
                target_percentage=85.0,
                measurement_window="7d",
                sli_metric="recommendation_accuracy",
                owner="ML Team",
                team="Data Science"
            )
        ]
        
        return {
            'streaming_slos': [slo.dict() for slo in music_streaming_slos],
            'total_slos': len(music_streaming_slos)
        }
    
    def create_example_configurations(self) -> Dict[str, Any]:
        """Créer exemples de configurations avancées"""
        
        examples = {
            'ml_drift_detection_example': {
                'description': 'Configuration détection drift pour modèles ML',
                'config': DriftDetection(
                    model_id="spotify_recommender_v3",
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity="high",
                    detection_method="statistical_distance",
                    drift_score=0.85,
                    confidence=0.92,
                    threshold_used=0.80,
                    affected_features=["user_engagement", "track_popularity"],
                    reference_period_start=datetime.utcnow() - timedelta(days=30),
                    reference_period_end=datetime.utcnow() - timedelta(days=7),
                    current_period_start=datetime.utcnow() - timedelta(days=7),
                    current_period_end=datetime.utcnow(),
                    recommended_actions=[
                        "Retrain model with recent data",
                        "Investigate feature importance changes",
                        "Update feature engineering pipeline"
                    ]
                ).dict()
            },
            
            'business_transaction_example': {
                'description': 'Transaction business pour monitoring',
                'config': BusinessTransaction(
                    transaction_type=TransactionType.RECOMMENDATION,
                    user_id="user_12345",
                    start_time=datetime.utcnow(),
                    duration_ms=150.0,
                    status="success",
                    steps_completed=5,
                    total_steps=5,
                    revenue_impact=0.25,
                    user_satisfaction_score=4.7,
                    service_calls=["user-service", "ml-service", "content-service"],
                    database_queries=3,
                    cache_hits=8,
                    cache_misses=2,
                    platform="mobile_ios",
                    country="US"
                ).dict()
            },
            
            'security_incident_example': {
                'description': 'Incident de sécurité avec investigation',
                'config': SecurityIncident(
                    title="Suspicious Music Content Access Pattern",
                    incident_type=ThreatType.DATA_EXFILTRATION,
                    severity=IncidentSeverity.SEV2,
                    description="Pattern inhabituel d'accès à contenu musical détecté",
                    summary="Utilisateur accédant à volume anormalement élevé de contenu",
                    detected_at=datetime.utcnow(),
                    reported_at=datetime.utcnow(),
                    affected_systems=["content-api", "streaming-service"],
                    affected_data=["music_metadata", "user_preferences"],
                    affected_users_count=1,
                    response_team=["security_analyst", "ml_engineer"],
                    actions_taken=[
                        "Temporary account restriction",
                        "Enhanced monitoring activated",
                        "ML model investigation initiated"
                    ],
                    created_by="security_system"
                ).dict()
            }
        }
        
        return examples
    
    def export_configuration_files(self, output_dir: str = "./spotify_config"):
        """Exporter tous les fichiers de configuration"""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration principale
        main_config = self.build_comprehensive_spotify_config()
        with open(output_path / "spotify_monitoring_config.json", "w") as f:
            json.dump(main_config, f, indent=2, default=str)
        
        # Exemples
        examples = self.create_example_configurations()
        with open(output_path / "configuration_examples.json", "w") as f:
            json.dump(examples, f, indent=2, default=str)
        
        # Scripts de déploiement
        deployment_scripts = self._generate_deployment_scripts()
        for script_name, script_content in deployment_scripts.items():
            with open(output_path / script_name, "w") as f:
                f.write(script_content)
        
        logger.info(f"Configuration files exported to {output_dir}")
    
    def _generate_deployment_scripts(self) -> Dict[str, str]:
        """Générer scripts de déploiement"""
        
        docker_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: spotify-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: spotify-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-clock-panel
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: spotify-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    restart: unless-stopped

  ml-monitoring:
    image: spotify/ml-monitoring:latest
    container_name: spotify-ml-monitoring
    ports:
      - "8080:8080"
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - ML_MODEL_REGISTRY_URL=http://mlflow:5000
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  grafana-storage:
        """
        
        makefile = """
# Spotify AI Agent Monitoring Makefile

.PHONY: deploy validate health test clean

# Deploy monitoring stack
deploy:
	@echo "🚀 Deploying Spotify monitoring stack..."
	docker-compose up -d
	@echo "✅ Monitoring stack deployed successfully!"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "🚨 AlertManager: http://localhost:9093"

# Validate configuration
validate:
	@echo "🔍 Validating monitoring configuration..."
	python monitoring_orchestrator.py --action=validate
	@echo "✅ Configuration validation completed!"

# Health check
health:
	@echo "🏥 Performing health check..."
	python monitoring_orchestrator.py --action=health
	@echo "✅ Health check completed!"

# Run tests
test:
	@echo "🧪 Running monitoring tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

# Generate configs
config:
	@echo "⚙️  Generating configurations..."
	python config_builder.py --build-spotify-config
	@echo "✅ Configurations generated!"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup completed!"

# Full setup
setup: config validate deploy health
	@echo "🎉 Spotify monitoring setup completed successfully!"
        """
        
        return {
            "docker-compose.yml": docker_compose,
            "Makefile": makefile
        }


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spotify Monitoring Configuration Builder')
    parser.add_argument('--build-spotify-config', action='store_true', 
                       help='Build comprehensive Spotify configuration')
    parser.add_argument('--create-examples', action='store_true',
                       help='Create configuration examples')
    parser.add_argument('--export-all', action='store_true',
                       help='Export all configuration files')
    parser.add_argument('--output-dir', default='./spotify_monitoring_config',
                       help='Output directory for configuration files')
    
    args = parser.parse_args()
    
    builder = SpotifyMonitoringConfigBuilder()
    
    if args.build_spotify_config:
        config = builder.build_comprehensive_spotify_config()
        print(json.dumps(config, indent=2, default=str))
    
    if args.create_examples:
        examples = builder.create_example_configurations()
        print(json.dumps(examples, indent=2, default=str))
    
    if args.export_all:
        builder.export_configuration_files(args.output_dir)
        print(f"✅ All configuration files exported to {args.output_dir}")


if __name__ == '__main__':
    main()
