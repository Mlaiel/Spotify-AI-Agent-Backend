"""
🧪 Integration Tests - Tenant Services
======================================

Comprehensive integration tests for tenant services including service-to-service
communication, database transactions, API endpoints, and workflow integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.tenancy.services import (
    TenantLifecycleService, TenantBillingService, 
    TenantAnalyticsService, TenantSecurityService,
    TenantMigrationService, TenantBackupService
)
from app.tenancy.models import (
    TenantModel, TenantDomain, TenantUser, TenantSubscription,
    TenantCreate, TenantUpdate, TenantResponse
)
from tests_backend.app.tenancy.fixtures.tenant_factories import (
    TenantFactory, TenantDataFactory, create_sample_tenant_data
)

pytestmark = pytest.mark.asyncio


class TestTenantLifecycleServiceIntegration:
    """🏢 Tenant Lifecycle Service Integration Tests"""
    
    @pytest.fixture
    async def lifecycle_service(self, db_session, redis_client):
        """Create TenantLifecycleService with real dependencies"""
        service = TenantLifecycleService()
        # Inject test dependencies
        service.db_session = db_session
        service.redis_client = redis_client
        yield service
        await service.cleanup()
    
    @pytest.fixture
    async def sample_tenant_data(self):
        """Create sample tenant data for integration tests"""
        return create_sample_tenant_data()
    
    async def test_complete_tenant_onboarding_workflow(
        self, lifecycle_service, sample_tenant_data
    ):
        """Test complete tenant onboarding workflow end-to-end"""
        # Step 1: Create tenant
        tenant_data = TenantCreate(**sample_tenant_data)
        created_tenant = await lifecycle_service.create_tenant(tenant_data)
        
        assert created_tenant.tenant_id is not None
        assert created_tenant.status == "pending"
        
        # Step 2: Setup infrastructure
        infrastructure_result = await lifecycle_service.setup_tenant_infrastructure(
            created_tenant.tenant_id
        )
        
        assert infrastructure_result["database"]["status"] == "created"
        assert infrastructure_result["cache"]["status"] == "configured"
        assert infrastructure_result["storage"]["status"] == "allocated"
        
        # Step 3: Configure domain and SSL
        domain_config = {
            "domain": "integration-test.example.com",
            "ssl_enabled": True,
            "auto_ssl": True
        }
        
        domain_result = await lifecycle_service.configure_tenant_domain(
            created_tenant.tenant_id, domain_config
        )
        
        assert domain_result["status"] == "configured"
        assert domain_result["ssl_certificate"]["status"] == "issued"
        
        # Step 4: Setup initial user and permissions
        admin_user = {
            "email": "admin@integration-test.example.com",
            "role": "admin",
            "permissions": ["full_access"]
        }
        
        user_result = await lifecycle_service.create_tenant_admin(
            created_tenant.tenant_id, admin_user
        )
        
        assert user_result["status"] == "created"
        assert user_result["user_id"] is not None
        
        # Step 5: Activate tenant
        activation_result = await lifecycle_service.activate_tenant(
            created_tenant.tenant_id
        )
        
        assert activation_result["status"] == "active"
        
        # Verify final state
        final_tenant = await lifecycle_service.get_tenant(created_tenant.tenant_id)
        assert final_tenant.status == "active"
        assert final_tenant.infrastructure_ready is True
        assert final_tenant.domain_configured is True
    
    async def test_tenant_upgrade_workflow(self, lifecycle_service):
        """Test tenant plan upgrade workflow"""
        # Create base tenant
        tenant_data = create_sample_tenant_data()
        tenant_data.update({"plan": "starter", "tier": "silver"})
        
        tenant = await lifecycle_service.create_tenant(TenantCreate(**tenant_data))
        await lifecycle_service.activate_tenant(tenant.tenant_id)
        
        # Upgrade to enterprise
        upgrade_data = {
            "new_plan": "enterprise",
            "new_tier": "platinum",
            "features_to_add": ["sso", "white_label", "priority_support"],
            "resource_updates": {
                "max_users": 5000,
                "max_storage_gb": 1000,
                "max_api_calls": 500000
            }
        }
        
        upgrade_result = await lifecycle_service.upgrade_tenant_plan(
            tenant.tenant_id, upgrade_data
        )
        
        assert upgrade_result["status"] == "upgraded"
        assert upgrade_result["new_plan"] == "enterprise"
        assert len(upgrade_result["enabled_features"]) == 3
        
        # Verify upgrade effects
        upgraded_tenant = await lifecycle_service.get_tenant(tenant.tenant_id)
        assert upgraded_tenant.plan == "enterprise"
        assert upgraded_tenant.max_users == 5000
        assert "sso" in upgraded_tenant.enabled_features
    
    async def test_tenant_suspension_and_reactivation(self, lifecycle_service):
        """Test tenant suspension and reactivation workflow"""
        # Create and activate tenant
        tenant_data = create_sample_tenant_data()
        tenant = await lifecycle_service.create_tenant(TenantCreate(**tenant_data))
        await lifecycle_service.activate_tenant(tenant.tenant_id)
        
        # Suspend tenant
        suspension_reason = "Payment overdue"
        suspension_result = await lifecycle_service.suspend_tenant(
            tenant.tenant_id, suspension_reason
        )
        
        assert suspension_result["status"] == "suspended"
        assert suspension_result["reason"] == suspension_reason
        
        # Verify suspension effects
        suspended_tenant = await lifecycle_service.get_tenant(tenant.tenant_id)
        assert suspended_tenant.status == "suspended"
        assert suspended_tenant.suspended_at is not None
        
        # Reactivate tenant
        reactivation_result = await lifecycle_service.reactivate_tenant(
            tenant.tenant_id
        )
        
        assert reactivation_result["status"] == "active"
        
        # Verify reactivation
        reactivated_tenant = await lifecycle_service.get_tenant(tenant.tenant_id)
        assert reactivated_tenant.status == "active"
        assert reactivated_tenant.reactivated_at is not None
    
    async def test_multi_tenant_isolation_verification(self, lifecycle_service):
        """Test multi-tenant data isolation during operations"""
        # Create two tenants
        tenant1_data = create_sample_tenant_data()
        tenant1_data.update({"name": "Tenant 1", "slug": "tenant-1"})
        tenant1 = await lifecycle_service.create_tenant(TenantCreate(**tenant1_data))
        
        tenant2_data = create_sample_tenant_data()
        tenant2_data.update({"name": "Tenant 2", "slug": "tenant-2"})
        tenant2 = await lifecycle_service.create_tenant(TenantCreate(**tenant2_data))
        
        # Activate both tenants
        await lifecycle_service.activate_tenant(tenant1.tenant_id)
        await lifecycle_service.activate_tenant(tenant2.tenant_id)
        
        # Create tenant-specific data
        tenant1_data_result = await lifecycle_service.create_tenant_data(
            tenant1.tenant_id, {"key": "tenant1_value", "secret": "tenant1_secret"}
        )
        
        tenant2_data_result = await lifecycle_service.create_tenant_data(
            tenant2.tenant_id, {"key": "tenant2_value", "secret": "tenant2_secret"}
        )
        
        # Verify isolation - tenant1 cannot access tenant2's data
        with pytest.raises(Exception, match="Access denied"):
            await lifecycle_service.access_tenant_data(
                tenant1.tenant_id, tenant2_data_result["data_id"]
            )
        
        # Verify isolation - tenant2 cannot access tenant1's data
        with pytest.raises(Exception, match="Access denied"):
            await lifecycle_service.access_tenant_data(
                tenant2.tenant_id, tenant1_data_result["data_id"]
            )
        
        # Verify each tenant can access their own data
        tenant1_retrieved = await lifecycle_service.access_tenant_data(
            tenant1.tenant_id, tenant1_data_result["data_id"]
        )
        assert tenant1_retrieved["key"] == "tenant1_value"
        
        tenant2_retrieved = await lifecycle_service.access_tenant_data(
            tenant2.tenant_id, tenant2_data_result["data_id"]
        )
        assert tenant2_retrieved["key"] == "tenant2_value"


class TestTenantBillingServiceIntegration:
    """💰 Tenant Billing Service Integration Tests"""
    
    @pytest.fixture
    async def billing_service(self, db_session):
        """Create TenantBillingService with dependencies"""
        service = TenantBillingService()
        service.db_session = db_session
        yield service
        await service.cleanup()
    
    async def test_complete_billing_cycle_workflow(self, billing_service):
        """Test complete billing cycle from usage to payment"""
        # Create tenant with billing plan
        tenant_id = "billing_integration_test"
        
        # Step 1: Setup subscription
        subscription_data = {
            "plan": "professional",
            "billing_cycle": "monthly",
            "start_date": datetime.utcnow(),
            "auto_renew": True
        }
        
        subscription = await billing_service.create_subscription(
            tenant_id, subscription_data
        )
        
        assert subscription["status"] == "active"
        assert subscription["subscription_id"] is not None
        
        # Step 2: Track usage over billing period
        usage_events = [
            {"resource": "api_calls", "amount": 15000, "timestamp": datetime.utcnow()},
            {"resource": "storage_gb", "amount": 250, "timestamp": datetime.utcnow()},
            {"resource": "bandwidth_gb", "amount": 500, "timestamp": datetime.utcnow()},
            {"resource": "compute_hours", "amount": 720, "timestamp": datetime.utcnow()}
        ]
        
        for event in usage_events:
            await billing_service.track_usage(tenant_id, event)
        
        # Step 3: Calculate monthly bill
        billing_period = {
            "start": datetime.utcnow().replace(day=1),
            "end": datetime.utcnow()
        }
        
        bill_calculation = await billing_service.calculate_monthly_bill(
            tenant_id, billing_period
        )
        
        assert bill_calculation["status"] == "calculated"
        assert bill_calculation["total_amount"] > 0
        assert "usage_charges" in bill_calculation
        assert "subscription_charges" in bill_calculation
        
        # Step 4: Generate invoice
        invoice = await billing_service.generate_invoice(tenant_id, bill_calculation)
        
        assert invoice["status"] == "generated"
        assert invoice["invoice_id"] is not None
        assert invoice["due_date"] is not None
        
        # Step 5: Process payment
        payment_data = {
            "invoice_id": invoice["invoice_id"],
            "payment_method": "test_card_token",
            "amount": bill_calculation["total_amount"]
        }
        
        payment_result = await billing_service.process_payment(payment_data)
        
        assert payment_result["status"] == "succeeded"
        assert payment_result["transaction_id"] is not None
        
        # Step 6: Update subscription status
        subscription_update = await billing_service.update_subscription_after_payment(
            tenant_id, payment_result
        )
        
        assert subscription_update["status"] == "current"
        assert subscription_update["next_billing_date"] is not None
    
    async def test_usage_aggregation_and_billing(self, billing_service):
        """Test usage aggregation across multiple services"""
        tenant_id = "usage_aggregation_test"
        
        # Simulate usage from multiple sources over time
        usage_data = [
            # API Service usage
            {"service": "api", "resource": "requests", "amount": 5000, "timestamp": datetime.utcnow() - timedelta(days=25)},
            {"service": "api", "resource": "requests", "amount": 4500, "timestamp": datetime.utcnow() - timedelta(days=20)},
            {"service": "api", "resource": "requests", "amount": 5500, "timestamp": datetime.utcnow() - timedelta(days=15)},
            
            # Storage Service usage
            {"service": "storage", "resource": "gb_hours", "amount": 100, "timestamp": datetime.utcnow() - timedelta(days=25)},
            {"service": "storage", "resource": "gb_hours", "amount": 120, "timestamp": datetime.utcnow() - timedelta(days=20)},
            {"service": "storage", "resource": "gb_hours", "amount": 150, "timestamp": datetime.utcnow() - timedelta(days=15)},
            
            # Compute Service usage
            {"service": "compute", "resource": "cpu_hours", "amount": 200, "timestamp": datetime.utcnow() - timedelta(days=25)},
            {"service": "compute", "resource": "cpu_hours", "amount": 250, "timestamp": datetime.utcnow() - timedelta(days=20)},
            {"service": "compute", "resource": "cpu_hours", "amount": 300, "timestamp": datetime.utcnow() - timedelta(days=15)}
        ]
        
        # Track all usage
        for usage in usage_data:
            await billing_service.track_service_usage(tenant_id, usage)
        
        # Aggregate usage for billing
        aggregation_result = await billing_service.aggregate_usage_for_billing(
            tenant_id, 
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        assert aggregation_result["status"] == "aggregated"
        assert aggregation_result["total_api_requests"] == 15000
        assert aggregation_result["total_storage_gb_hours"] == 370
        assert aggregation_result["total_cpu_hours"] == 750
        
        # Calculate costs
        cost_calculation = await billing_service.calculate_aggregated_costs(
            tenant_id, aggregation_result
        )
        
        assert cost_calculation["api_costs"] > 0
        assert cost_calculation["storage_costs"] > 0
        assert cost_calculation["compute_costs"] > 0
        assert cost_calculation["total_costs"] > 0
    
    async def test_billing_dispute_and_adjustment_workflow(self, billing_service):
        """Test billing dispute handling and adjustments"""
        tenant_id = "billing_dispute_test"
        
        # Create an invoice
        invoice_data = {
            "tenant_id": tenant_id,
            "amount": Decimal("500.00"),
            "line_items": [
                {"description": "Base subscription", "amount": Decimal("299.99")},
                {"description": "API overage", "amount": Decimal("200.01")}
            ]
        }
        
        invoice = await billing_service.create_invoice(invoice_data)
        
        # Submit dispute
        dispute_data = {
            "invoice_id": invoice["invoice_id"],
            "disputed_amount": Decimal("200.01"),
            "reason": "API usage calculation error",
            "evidence": "Usage logs show only 10,000 API calls, not 25,000 as billed"
        }
        
        dispute_result = await billing_service.submit_billing_dispute(dispute_data)
        
        assert dispute_result["status"] == "under_review"
        assert dispute_result["dispute_id"] is not None
        
        # Process dispute (simulate approval)
        dispute_resolution = {
            "dispute_id": dispute_result["dispute_id"],
            "resolution": "approved",
            "adjustment_amount": Decimal("-150.01"),  # Partial credit
            "resolution_notes": "Usage recalculated, partial credit issued"
        }
        
        resolution_result = await billing_service.resolve_billing_dispute(
            dispute_resolution
        )
        
        assert resolution_result["status"] == "resolved"
        assert resolution_result["adjustment_applied"] is True
        
        # Verify adjustment reflected in billing
        adjusted_invoice = await billing_service.get_invoice(invoice["invoice_id"])
        assert adjusted_invoice["adjusted_amount"] == Decimal("349.99")
        assert len(adjusted_invoice["adjustments"]) == 1


class TestTenantAnalyticsServiceIntegration:
    """📊 Tenant Analytics Service Integration Tests"""
    
    @pytest.fixture
    async def analytics_service(self, db_session):
        """Create TenantAnalyticsService with dependencies"""
        service = TenantAnalyticsService()
        service.db_session = db_session
        yield service
        await service.cleanup()
    
    async def test_real_time_analytics_pipeline(self, analytics_service):
        """Test real-time analytics data pipeline"""
        tenant_id = "analytics_pipeline_test"
        
        # Step 1: Setup analytics collection
        collection_config = {
            "metrics": ["user_activity", "api_usage", "performance", "errors"],
            "collection_interval": 30,  # seconds
            "retention_period": 90  # days
        }
        
        setup_result = await analytics_service.setup_analytics_collection(
            tenant_id, collection_config
        )
        
        assert setup_result["status"] == "configured"
        assert len(setup_result["configured_metrics"]) == 4
        
        # Step 2: Simulate real-time events
        events = [
            {"type": "user_login", "user_id": "user_123", "timestamp": datetime.utcnow()},
            {"type": "api_request", "endpoint": "/api/v1/users", "response_time": 45, "timestamp": datetime.utcnow()},
            {"type": "api_request", "endpoint": "/api/v1/products", "response_time": 67, "timestamp": datetime.utcnow()},
            {"type": "error", "error_code": "404", "endpoint": "/api/v1/missing", "timestamp": datetime.utcnow()},
            {"type": "user_action", "action": "create_product", "user_id": "user_123", "timestamp": datetime.utcnow()}
        ]
        
        # Send events to analytics pipeline
        for event in events:
            await analytics_service.track_event(tenant_id, event)
        
        # Step 3: Process events and generate metrics
        processing_result = await analytics_service.process_events_batch(tenant_id)
        
        assert processing_result["events_processed"] == 5
        assert processing_result["metrics_generated"] > 0
        
        # Step 4: Query real-time metrics
        current_metrics = await analytics_service.get_current_metrics(tenant_id)
        
        assert "active_users" in current_metrics
        assert "api_requests_per_minute" in current_metrics
        assert "average_response_time" in current_metrics
        assert "error_rate" in current_metrics
        
        # Step 5: Generate analytics dashboard data
        dashboard_data = await analytics_service.generate_dashboard_data(tenant_id)
        
        assert "summary_stats" in dashboard_data
        assert "time_series_data" in dashboard_data
        assert "top_endpoints" in dashboard_data
        assert "user_activity_heatmap" in dashboard_data
    
    async def test_historical_analytics_reporting(self, analytics_service):
        """Test historical analytics and trend analysis"""
        tenant_id = "historical_analytics_test"
        
        # Generate historical data (simulate 30 days of activity)
        historical_data = []
        base_date = datetime.utcnow() - timedelta(days=30)
        
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            # Simulate daily metrics with growth trend
            daily_metrics = {
                "date": current_date.date(),
                "active_users": 100 + (day * 2),  # Growing user base
                "api_calls": 5000 + (day * 100),  # Increasing API usage
                "response_time_avg": 50 + (day * 0.5),  # Slight performance degradation
                "error_rate": max(0.01, 0.05 - (day * 0.001)),  # Improving error rate
                "revenue": 1000 + (day * 50)  # Growing revenue
            }
            
            historical_data.append(daily_metrics)
        
        # Store historical data
        for data in historical_data:
            await analytics_service.store_daily_metrics(tenant_id, data)
        
        # Generate trend analysis report
        trend_config = {
            "period": "30_days",
            "metrics": ["active_users", "api_calls", "response_time_avg", "error_rate"],
            "include_forecasting": True
        }
        
        trend_report = await analytics_service.generate_trend_analysis(
            tenant_id, trend_config
        )
        
        assert trend_report["status"] == "generated"
        assert "growth_rates" in trend_report
        assert "trend_direction" in trend_report
        assert "forecasts" in trend_report
        
        # Verify growth trends
        assert trend_report["growth_rates"]["active_users"] > 0  # Growing
        assert trend_report["growth_rates"]["api_calls"] > 0  # Growing
        assert trend_report["trend_direction"]["error_rate"] == "decreasing"  # Improving
    
    async def test_custom_analytics_dashboard_creation(self, analytics_service):
        """Test custom analytics dashboard creation and configuration"""
        tenant_id = "custom_dashboard_test"
        
        # Define custom dashboard configuration
        dashboard_config = {
            "name": "Executive Dashboard",
            "widgets": [
                {
                    "type": "metric_card",
                    "metric": "monthly_revenue",
                    "title": "Monthly Revenue",
                    "format": "currency"
                },
                {
                    "type": "line_chart",
                    "metrics": ["active_users", "new_signups"],
                    "title": "User Growth",
                    "time_range": "30_days"
                },
                {
                    "type": "bar_chart",
                    "metric": "api_calls_by_endpoint",
                    "title": "Top API Endpoints",
                    "limit": 10
                },
                {
                    "type": "gauge",
                    "metric": "system_health_score",
                    "title": "System Health",
                    "thresholds": {"warning": 70, "critical": 50}
                }
            ],
            "refresh_interval": 300,  # 5 minutes
            "permissions": ["admin", "manager"]
        }
        
        # Create custom dashboard
        dashboard_result = await analytics_service.create_custom_dashboard(
            tenant_id, dashboard_config
        )
        
        assert dashboard_result["status"] == "created"
        assert dashboard_result["dashboard_id"] is not None
        assert len(dashboard_result["configured_widgets"]) == 4
        
        # Generate dashboard data
        dashboard_data = await analytics_service.generate_dashboard_data(
            tenant_id, dashboard_result["dashboard_id"]
        )
        
        assert "widgets_data" in dashboard_data
        assert len(dashboard_data["widgets_data"]) == 4
        
        # Test dashboard sharing
        share_config = {
            "dashboard_id": dashboard_result["dashboard_id"],
            "share_with": ["user@company.com"],
            "permissions": ["view"],
            "expiry_date": datetime.utcnow() + timedelta(days=30)
        }
        
        share_result = await analytics_service.share_dashboard(share_config)
        
        assert share_result["status"] == "shared"
        assert share_result["share_url"] is not None


class TestTenantSecurityServiceIntegration:
    """🔒 Tenant Security Service Integration Tests"""
    
    @pytest.fixture
    async def security_service(self, db_session):
        """Create TenantSecurityService with dependencies"""
        service = TenantSecurityService()
        service.db_session = db_session
        yield service
        await service.cleanup()
    
    async def test_security_audit_workflow(self, security_service):
        """Test comprehensive security audit workflow"""
        tenant_id = "security_audit_test"
        
        # Step 1: Initialize security audit
        audit_config = {
            "scope": "comprehensive",
            "include_penetration_testing": True,
            "include_vulnerability_scanning": True,
            "include_compliance_check": True,
            "standards": ["OWASP", "NIST", "ISO27001"]
        }
        
        audit_initiation = await security_service.initiate_security_audit(
            tenant_id, audit_config
        )
        
        assert audit_initiation["status"] == "initiated"
        assert audit_initiation["audit_id"] is not None
        
        # Step 2: Run vulnerability scan
        vuln_scan_result = await security_service.run_vulnerability_scan(
            tenant_id, audit_initiation["audit_id"]
        )
        
        assert vuln_scan_result["status"] == "completed"
        assert "vulnerabilities_found" in vuln_scan_result
        assert "risk_score" in vuln_scan_result
        
        # Step 3: Run penetration tests
        pen_test_result = await security_service.run_penetration_tests(
            tenant_id, audit_initiation["audit_id"]
        )
        
        assert pen_test_result["status"] == "completed"
        assert "test_results" in pen_test_result
        assert "security_weaknesses" in pen_test_result
        
        # Step 4: Check compliance
        compliance_result = await security_service.check_compliance(
            tenant_id, audit_config["standards"]
        )
        
        assert compliance_result["status"] == "completed"
        assert "compliance_score" in compliance_result
        assert len(compliance_result["standards_checked"]) == 3
        
        # Step 5: Generate security report
        report_result = await security_service.generate_security_report(
            tenant_id, audit_initiation["audit_id"]
        )
        
        assert report_result["status"] == "generated"
        assert report_result["report_id"] is not None
        assert "executive_summary" in report_result
        assert "detailed_findings" in report_result
        assert "remediation_plan" in report_result
    
    async def test_incident_response_workflow(self, security_service):
        """Test security incident response workflow"""
        tenant_id = "incident_response_test"
        
        # Step 1: Detect security incident
        incident_data = {
            "type": "suspicious_login_activity",
            "severity": "high",
            "description": "Multiple failed login attempts from unknown IP addresses",
            "affected_resources": ["user_authentication_service"],
            "detection_time": datetime.utcnow(),
            "source_ip": "192.168.1.100",
            "user_affected": "admin@tenant.com"
        }
        
        incident_creation = await security_service.create_security_incident(
            tenant_id, incident_data
        )
        
        assert incident_creation["status"] == "created"
        assert incident_creation["incident_id"] is not None
        
        # Step 2: Analyze incident
        analysis_result = await security_service.analyze_security_incident(
            tenant_id, incident_creation["incident_id"]
        )
        
        assert analysis_result["status"] == "analyzed"
        assert "threat_level" in analysis_result
        assert "potential_impact" in analysis_result
        assert "recommended_actions" in analysis_result
        
        # Step 3: Implement containment measures
        containment_actions = [
            {"action": "block_ip_address", "target": "192.168.1.100"},
            {"action": "force_password_reset", "target": "admin@tenant.com"},
            {"action": "enable_mfa", "target": "all_admin_users"},
            {"action": "increase_monitoring", "target": "authentication_service"}
        ]
        
        containment_result = await security_service.implement_containment(
            tenant_id, incident_creation["incident_id"], containment_actions
        )
        
        assert containment_result["status"] == "implemented"
        assert len(containment_result["actions_taken"]) == 4
        
        # Step 4: Generate incident report
        incident_report = await security_service.generate_incident_report(
            tenant_id, incident_creation["incident_id"]
        )
        
        assert incident_report["status"] == "generated"
        assert "timeline" in incident_report
        assert "root_cause_analysis" in incident_report
        assert "lessons_learned" in incident_report


# Shared fixtures for integration tests
@pytest.fixture(scope="session")
async def test_database():
    """Create test database for integration tests"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.tenancy.models import Base
    
    # Create test database engine
    engine = create_engine("sqlite:///test_integration.db", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield SessionLocal
    
    # Cleanup
    Base.metadata.drop_all(engine)


@pytest.fixture
async def db_session(test_database):
    """Create database session for tests"""
    session = test_database()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
async def redis_client():
    """Create Redis client for tests"""
    import fakeredis.aioredis
    
    redis = fakeredis.aioredis.FakeRedis()
    yield redis
    await redis.close()


@pytest.fixture
async def http_client():
    """Create HTTP client for API tests"""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def mock_external_services():
    """Mock external service dependencies"""
    with patch('app.tenancy.external.stripe_client') as mock_stripe, \
         patch('app.tenancy.external.sendgrid_client') as mock_email, \
         patch('app.tenancy.external.kubernetes_client') as mock_k8s:
        
        # Configure mocks
        mock_stripe.create_subscription.return_value = {"id": "sub_test", "status": "active"}
        mock_email.send_email.return_value = {"status": "sent", "message_id": "msg_test"}
        mock_k8s.create_namespace.return_value = {"status": "created"}
        
        yield {
            "stripe": mock_stripe,
            "email": mock_email,
            "kubernetes": mock_k8s
        }
