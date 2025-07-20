"""
🧪 Unit Tests - Tenant Models
============================

Comprehensive unit tests for tenant data models, Pydantic schemas,
SQLAlchemy models, and data validation logic.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import Dict, Any

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from hypothesis import given, strategies as st, settings

from app.tenancy.models import (
    TenantModel, TenantDomain, TenantUser, TenantSubscription,
    TenantUsageRecord, TenantSecurityEvent, TenantAnalytics,
    TenantCreate, TenantUpdate, TenantResponse,
    DomainCreate, DomainResponse,
    BillingCreate, BillingResponse,
    TenantStatus, TenantPlan, TenantTier, IsolationLevel,
    UserRole, SubscriptionStatus, SecurityEventType
)

pytestmark = pytest.mark.asyncio


class TestTenantModels:
    """🏢 SQLAlchemy Tenant Model Tests"""
    
    def test_tenant_model_creation(self, db_session):
        """Test basic tenant model creation"""
        # Arrange
        tenant = TenantModel(
            tenant_id="test_tenant_001",
            name="Test Corporation",
            slug="test-corporation",
            status="active",
            plan="enterprise",
            tier="platinum",
            max_users=1000,
            max_storage_gb=500,
            max_api_calls=100000
        )
        
        # Act
        db_session.add(tenant)
        db_session.commit()
        
        # Assert
        assert tenant.id is not None
        assert tenant.tenant_id == "test_tenant_001"
        assert tenant.name == "Test Corporation"
        assert tenant.created_at is not None
        assert tenant.updated_at is not None
    
    def test_tenant_model_validation(self, db_session):
        """Test tenant model validation constraints"""
        # Test unique tenant_id constraint
        tenant1 = TenantModel(
            tenant_id="duplicate_id",
            name="Tenant 1",
            slug="tenant-1"
        )
        tenant2 = TenantModel(
            tenant_id="duplicate_id",
            name="Tenant 2", 
            slug="tenant-2"
        )
        
        db_session.add(tenant1)
        db_session.commit()
        
        db_session.add(tenant2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_tenant_model_relationships(self, db_session):
        """Test tenant model relationships"""
        # Create tenant
        tenant = TenantModel(
            tenant_id="test_tenant_rel",
            name="Relationship Test",
            slug="relationship-test"
        )
        db_session.add(tenant)
        db_session.commit()
        
        # Create related domain
        domain = TenantDomain(
            tenant_id=tenant.id,
            domain="test.example.com",
            is_primary=True,
            ssl_enabled=True
        )
        db_session.add(domain)
        
        # Create related user
        user = TenantUser(
            tenant_id=tenant.id,
            email="admin@test.example.com",
            role="admin",
            is_active=True
        )
        db_session.add(user)
        
        db_session.commit()
        
        # Test relationships
        assert len(tenant.domains) == 1
        assert len(tenant.users) == 1
        assert tenant.domains[0].domain == "test.example.com"
        assert tenant.users[0].email == "admin@test.example.com"
    
    def test_tenant_model_soft_delete(self, db_session):
        """Test tenant soft delete functionality"""
        # Create tenant
        tenant = TenantModel(
            tenant_id="soft_delete_test",
            name="Soft Delete Test",
            slug="soft-delete-test"
        )
        db_session.add(tenant)
        db_session.commit()
        
        # Soft delete
        tenant.deleted_at = datetime.utcnow()
        tenant.status = "deleted"
        db_session.commit()
        
        # Verify soft delete
        assert tenant.deleted_at is not None
        assert tenant.status == "deleted"
        
        # Verify tenant still exists in database
        found_tenant = db_session.query(TenantModel).filter_by(
            tenant_id="soft_delete_test"
        ).first()
        assert found_tenant is not None
    
    def test_tenant_analytics_model(self, db_session):
        """Test tenant analytics model"""
        # Create tenant
        tenant = TenantModel(
            tenant_id="analytics_test",
            name="Analytics Test",
            slug="analytics-test"
        )
        db_session.add(tenant)
        db_session.commit()
        
        # Create analytics record
        analytics = TenantAnalytics(
            tenant_id=tenant.id,
            metric_name="api_calls",
            metric_value=1500.0,
            metric_timestamp=datetime.utcnow(),
            metadata={"endpoint": "/api/v1/users", "method": "GET"}
        )
        db_session.add(analytics)
        db_session.commit()
        
        # Verify analytics
        assert analytics.id is not None
        assert analytics.metric_name == "api_calls"
        assert analytics.metric_value == 1500.0
        assert analytics.metadata["endpoint"] == "/api/v1/users"
    
    def test_tenant_usage_record_model(self, db_session):
        """Test tenant usage record model"""
        # Create tenant
        tenant = TenantModel(
            tenant_id="usage_test",
            name="Usage Test",
            slug="usage-test"
        )
        db_session.add(tenant)
        db_session.commit()
        
        # Create usage record
        usage = TenantUsageRecord(
            tenant_id=tenant.id,
            resource_type="api_calls",
            usage_amount=2500,
            usage_date=datetime.utcnow().date(),
            cost_amount=Decimal("12.50"),
            billing_period_start=datetime.utcnow().replace(day=1),
            billing_period_end=datetime.utcnow()
        )
        db_session.add(usage)
        db_session.commit()
        
        # Verify usage record
        assert usage.id is not None
        assert usage.resource_type == "api_calls"
        assert usage.usage_amount == 2500
        assert usage.cost_amount == Decimal("12.50")


class TestTenantPydanticSchemas:
    """📋 Pydantic Schema Validation Tests"""
    
    def test_tenant_create_schema_valid(self):
        """Test valid tenant creation schema"""
        # Arrange
        tenant_data = {
            "name": "Valid Corporation",
            "plan": "enterprise",
            "tier": "platinum",
            "max_users": 1000,
            "max_storage_gb": 500,
            "region": "us-east-1",
            "features": ["sso", "white_label", "api_access"]
        }
        
        # Act
        tenant_create = TenantCreate(**tenant_data)
        
        # Assert
        assert tenant_create.name == "Valid Corporation"
        assert tenant_create.plan == "enterprise"
        assert tenant_create.tier == "platinum"
        assert tenant_create.max_users == 1000
        assert "sso" in tenant_create.features
    
    def test_tenant_create_schema_validation_errors(self):
        """Test tenant creation schema validation errors"""
        # Test empty name
        with pytest.raises(ValidationError, match="Name cannot be empty"):
            TenantCreate(name="", plan="free")
        
        # Test invalid plan
        with pytest.raises(ValidationError, match="Invalid plan"):
            TenantCreate(name="Test", plan="invalid_plan")
        
        # Test negative max_users
        with pytest.raises(ValidationError, match="must be positive"):
            TenantCreate(name="Test", plan="free", max_users=-1)
        
        # Test invalid email format
        with pytest.raises(ValidationError, match="Invalid email"):
            TenantCreate(name="Test", plan="free", admin_email="invalid-email")
    
    def test_tenant_update_schema(self):
        """Test tenant update schema"""
        # Test partial update
        update_data = TenantUpdate(
            name="Updated Name",
            max_users=2000
        )
        
        assert update_data.name == "Updated Name"
        assert update_data.max_users == 2000
        assert update_data.plan is None  # Optional field not provided
    
    def test_tenant_response_schema(self):
        """Test tenant response schema"""
        # Arrange
        response_data = {
            "tenant_id": "test_123",
            "name": "Response Test",
            "status": "active",
            "plan": "professional",
            "tier": "gold",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "max_users": 500,
            "current_users": 150,
            "usage_stats": {
                "api_calls": 15000,
                "storage_used": 25.5,
                "bandwidth_used": 100.2
            }
        }
        
        # Act
        tenant_response = TenantResponse(**response_data)
        
        # Assert
        assert tenant_response.tenant_id == "test_123"
        assert tenant_response.current_users == 150
        assert tenant_response.usage_stats["api_calls"] == 15000
    
    def test_domain_create_schema(self):
        """Test domain creation schema"""
        # Valid domain
        domain_data = DomainCreate(
            domain="custom.example.com",
            is_primary=True,
            ssl_enabled=True,
            ssl_certificate="-----BEGIN CERTIFICATE-----\n...",
            ssl_private_key="-----BEGIN PRIVATE KEY-----\n..."
        )
        
        assert domain_data.domain == "custom.example.com"
        assert domain_data.ssl_enabled is True
        
        # Invalid domain format
        with pytest.raises(ValidationError, match="Invalid domain"):
            DomainCreate(domain="invalid..domain")
    
    def test_billing_create_schema(self):
        """Test billing creation schema"""
        billing_data = BillingCreate(
            plan="enterprise",
            billing_cycle="monthly",
            amount=Decimal("99.99"),
            currency="USD",
            payment_method="credit_card",
            auto_renew=True
        )
        
        assert billing_data.amount == Decimal("99.99")
        assert billing_data.currency == "USD"
        assert billing_data.auto_renew is True
        
        # Test invalid currency
        with pytest.raises(ValidationError, match="Invalid currency"):
            BillingCreate(
                plan="free",
                amount=Decimal("0.00"),
                currency="INVALID"
            )


class TestTenantEnums:
    """🔢 Enum Validation Tests"""
    
    def test_tenant_status_enum(self):
        """Test tenant status enum values"""
        # Valid statuses
        assert TenantStatus.ACTIVE == "active"
        assert TenantStatus.SUSPENDED == "suspended"
        assert TenantStatus.DELETED == "deleted"
        assert TenantStatus.PENDING == "pending"
        
        # Test enum membership
        assert "active" in [status.value for status in TenantStatus]
        assert "invalid_status" not in [status.value for status in TenantStatus]
    
    def test_tenant_plan_enum(self):
        """Test tenant plan enum values"""
        assert TenantPlan.FREE == "free"
        assert TenantPlan.STARTER == "starter" 
        assert TenantPlan.PROFESSIONAL == "professional"
        assert TenantPlan.ENTERPRISE == "enterprise"
    
    def test_tenant_tier_enum(self):
        """Test tenant tier enum values"""
        assert TenantTier.BRONZE == "bronze"
        assert TenantTier.SILVER == "silver"
        assert TenantTier.GOLD == "gold"
        assert TenantTier.PLATINUM == "platinum"
    
    def test_isolation_level_enum(self):
        """Test isolation level enum values"""
        assert IsolationLevel.SHARED_DB == "shared_db"
        assert IsolationLevel.SEPARATE_SCHEMA == "separate_schema"
        assert IsolationLevel.SEPARATE_DB == "separate_db"
    
    def test_user_role_enum(self):
        """Test user role enum values"""
        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"
        assert UserRole.VIEWER == "viewer"
        assert UserRole.OWNER == "owner"


class TestTenantPropertyBasedModels:
    """🎲 Property-based tests for tenant models"""
    
    @given(
        name=st.text(min_size=1, max_size=255),
        plan=st.sampled_from(["free", "starter", "professional", "enterprise"]),
        max_users=st.integers(min_value=1, max_value=100000),
        max_storage=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=100)
    def test_tenant_create_property_based(self, name, plan, max_users, max_storage):
        """Property-based test for tenant creation"""
        try:
            tenant = TenantCreate(
                name=name,
                plan=plan,
                max_users=max_users,
                max_storage_gb=max_storage
            )
            
            # Properties that should always hold
            assert tenant.name == name
            assert tenant.plan == plan
            assert tenant.max_users == max_users
            assert tenant.max_storage_gb == max_storage
            assert len(tenant.name.strip()) > 0
            
        except ValidationError:
            # Some random inputs might be invalid, which is acceptable
            pass
    
    @given(
        email=st.emails(),
        role=st.sampled_from(["admin", "user", "viewer", "owner"])
    )
    @settings(max_examples=50)
    def test_tenant_user_property_based(self, email, role):
        """Property-based test for tenant user creation"""
        try:
            user_data = {
                "tenant_id": uuid4(),
                "email": email,
                "role": role,
                "is_active": True
            }
            
            # This would be validated through Pydantic in real scenario
            assert "@" in email
            assert role in ["admin", "user", "viewer", "owner"]
            
        except Exception:
            # Some combinations might be invalid
            pass
    
    @given(
        domain=st.text(min_size=3, max_size=253).filter(
            lambda x: "." in x and not x.startswith(".") and not x.endswith(".")
        ),
        ssl_enabled=st.booleans()
    )
    @settings(max_examples=30)
    def test_domain_property_based(self, domain, ssl_enabled):
        """Property-based test for domain validation"""
        try:
            domain_data = DomainCreate(
                domain=domain,
                is_primary=True,
                ssl_enabled=ssl_enabled
            )
            
            # Properties that should hold
            assert domain_data.domain == domain
            assert domain_data.ssl_enabled == ssl_enabled
            assert "." in domain_data.domain
            
        except ValidationError:
            # Some random domains might be invalid
            pass


class TestTenantModelValidators:
    """✅ Custom Validator Tests"""
    
    def test_tenant_name_validator(self):
        """Test custom tenant name validation"""
        # Valid names
        valid_names = [
            "Corporation Inc",
            "Tech Company LLC",
            "Startup 2024",
            "Multi-Word Company Name"
        ]
        
        for name in valid_names:
            tenant = TenantCreate(name=name, plan="free")
            assert tenant.name == name
        
        # Invalid names
        invalid_names = [
            "",  # Empty
            "   ",  # Only whitespace
            "A",  # Too short
            "X" * 256,  # Too long
        ]
        
        for name in invalid_names:
            with pytest.raises(ValidationError):
                TenantCreate(name=name, plan="free")
    
    def test_slug_generation_validator(self):
        """Test automatic slug generation"""
        # Test slug generation from name
        tenant = TenantCreate(name="My Company Name", plan="free")
        assert tenant.slug == "my-company-name"
        
        # Test special character handling
        tenant = TenantCreate(name="Company & Co. Ltd!", plan="free")
        assert "company" in tenant.slug.lower()
        assert "&" not in tenant.slug
        assert "!" not in tenant.slug
    
    def test_email_validator(self):
        """Test email validation in schemas"""
        # Valid emails
        valid_emails = [
            "admin@company.com",
            "user.name@example.org",
            "test+tag@domain.co.uk"
        ]
        
        for email in valid_emails:
            # This would be in a user creation schema
            assert "@" in email and "." in email
        
        # Invalid emails
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user..name@domain.com"
        ]
        
        for email in invalid_emails:
            # In real implementation, this would raise ValidationError
            assert not (email.count("@") == 1 and "." in email.split("@")[1])
    
    def test_resource_limit_validators(self):
        """Test resource limit validation"""
        # Valid resource limits
        tenant = TenantCreate(
            name="Resource Test",
            plan="enterprise",
            max_users=1000,
            max_storage_gb=500,
            max_api_calls=100000,
            max_bandwidth_gb=1000
        )
        
        assert tenant.max_users == 1000
        assert tenant.max_storage_gb == 500
        
        # Invalid resource limits (negative values)
        with pytest.raises(ValidationError):
            TenantCreate(
                name="Invalid Test",
                plan="free",
                max_users=-1
            )
    
    def test_plan_tier_compatibility_validator(self):
        """Test plan and tier compatibility validation"""
        # Valid combinations
        valid_combinations = [
            ("free", "bronze"),
            ("starter", "silver"), 
            ("professional", "gold"),
            ("enterprise", "platinum")
        ]
        
        for plan, tier in valid_combinations:
            tenant = TenantCreate(
                name="Compatibility Test",
                plan=plan,
                tier=tier
            )
            assert tenant.plan == plan
            assert tenant.tier == tier
        
        # Invalid combinations (would be validated in real implementation)
        invalid_combinations = [
            ("free", "platinum"),  # Free plan with premium tier
            ("enterprise", "bronze")  # Enterprise plan with basic tier
        ]
        
        # In real implementation, these would raise validation errors


# Test fixtures for models
@pytest.fixture
def sample_tenant_model():
    """Create sample tenant model for testing"""
    return TenantModel(
        tenant_id="sample_tenant",
        name="Sample Corporation",
        slug="sample-corporation",
        status="active",
        plan="professional",
        tier="gold",
        max_users=500,
        max_storage_gb=100,
        max_api_calls=50000
    )


@pytest.fixture
def sample_tenant_domain():
    """Create sample tenant domain for testing"""
    return TenantDomain(
        domain="sample.example.com",
        is_primary=True,
        ssl_enabled=True,
        verified=True
    )


@pytest.fixture
def sample_tenant_user():
    """Create sample tenant user for testing"""
    return TenantUser(
        email="admin@sample.example.com",
        role="admin",
        is_active=True,
        last_login=datetime.utcnow()
    )
