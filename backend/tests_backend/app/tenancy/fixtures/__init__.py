"""
🔧 Test Fixtures & Utilities - Comprehensive Testing Infrastructure
================================================================

Advanced test fixtures, factories, and utilities for enterprise-grade
tenancy testing with realistic data generation and test environment setup.
"""

import pytest
import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import json
import faker
from dataclasses import dataclass, field

from app.tenancy.models import TenantCreate, TenantUser, TenantUpdate
from app.tenancy import EnterpriseTenantManager
from app.tenancy.advanced_managers import TenantSecurityManager, QuantumDataIsolationManager
from app.tenancy.services import TenantAnalyticsService, ComplianceService

# Initialize Faker for realistic test data
fake = faker.Faker(['en_US', 'fr_FR', 'de_DE', 'es_ES', 'it_IT'])


@dataclass
class TenantTestProfile:
    """Comprehensive tenant test profile for different scenarios"""
    tenant_type: str
    plan_tier: str
    user_count: int
    data_volume: str
    compliance_requirements: List[str]
    security_level: str
    geographic_region: str
    industry: str
    special_features: List[str] = field(default_factory=list)
    test_scenario: str = ""


class TenantDataFactory:
    """🏭 Factory for generating realistic tenant test data"""
    
    @staticmethod
    def create_sample_tenant_data(profile: Optional[TenantTestProfile] = None) -> Dict[str, Any]:
        """Generate realistic tenant data based on profile"""
        if not profile:
            profile = TenantTestProfile(
                tenant_type="business",
                plan_tier="professional",
                user_count=random.randint(10, 100),
                data_volume="medium",
                compliance_requirements=["GDPR"],
                security_level="standard",
                geographic_region="EU",
                industry="technology"
            )
        
        company_name = fake.company()
        
        tenant_data = {
            "name": f"{company_name} {profile.tenant_type.title()}",
            "description": f"Enterprise tenant for {company_name} - {profile.industry} industry",
            "plan": profile.plan_tier,
            "status": "active",
            "tenant_type": profile.tenant_type,
            "metadata": {
                "industry": profile.industry,
                "company_size": TenantDataFactory._get_company_size(profile.user_count),
                "geographic_region": profile.geographic_region,
                "compliance_requirements": profile.compliance_requirements,
                "security_level": profile.security_level,
                "data_volume": profile.data_volume,
                "contact_email": fake.company_email(),
                "phone": fake.phone_number(),
                "address": {
                    "street": fake.street_address(),
                    "city": fake.city(),
                    "state": fake.state(),
                    "postal_code": fake.postcode(),
                    "country": TenantDataFactory._get_country_for_region(profile.geographic_region)
                },
                "billing_info": {
                    "billing_email": fake.email(),
                    "payment_method": random.choice(["credit_card", "bank_transfer", "invoice"]),
                    "billing_cycle": random.choice(["monthly", "quarterly", "annual"]),
                    "currency": TenantDataFactory._get_currency_for_region(profile.geographic_region)
                },
                "technical_specs": {
                    "max_users": profile.user_count * 2,  # Allow for growth
                    "storage_limit_gb": TenantDataFactory._get_storage_limit(profile.data_volume),
                    "api_rate_limit": TenantDataFactory._get_api_rate_limit(profile.plan_tier),
                    "backup_retention_days": TenantDataFactory._get_backup_retention(profile.plan_tier),
                    "sla_tier": TenantDataFactory._get_sla_tier(profile.plan_tier)
                },
                "feature_flags": {
                    "advanced_analytics": profile.plan_tier in ["professional", "enterprise"],
                    "real_time_sync": profile.plan_tier == "enterprise",
                    "custom_integrations": profile.plan_tier == "enterprise",
                    "white_labeling": profile.plan_tier == "enterprise",
                    "audit_logging": profile.security_level in ["high", "enterprise"],
                    "data_encryption": True,
                    "compliance_reporting": len(profile.compliance_requirements) > 0
                },
                "integration_settings": {
                    "webhooks_enabled": True,
                    "api_access": True,
                    "third_party_integrations": random.sample([
                        "salesforce", "hubspot", "slack", "teams", "zapier", 
                        "google_workspace", "office365", "jira", "confluence"
                    ], k=random.randint(2, 5)),
                    "sso_provider": random.choice(["okta", "auth0", "azure_ad", "google", "custom"]) if profile.security_level in ["high", "enterprise"] else None
                }
            },
            "configuration": {
                "timezone": TenantDataFactory._get_timezone_for_region(profile.geographic_region),
                "locale": TenantDataFactory._get_locale_for_region(profile.geographic_region),
                "date_format": TenantDataFactory._get_date_format_for_region(profile.geographic_region),
                "currency_format": TenantDataFactory._get_currency_format_for_region(profile.geographic_region),
                "data_retention_policy": {
                    "active_data_days": 365,
                    "archived_data_years": 7,
                    "deletion_policy": "automatic" if "GDPR" in profile.compliance_requirements else "manual"
                },
                "privacy_settings": {
                    "data_processing_consent": True,
                    "marketing_consent": random.choice([True, False]),
                    "analytics_consent": True,
                    "cookie_consent": True,
                    "gdpr_compliant": "GDPR" in profile.compliance_requirements
                },
                "notification_preferences": {
                    "email_notifications": True,
                    "sms_notifications": random.choice([True, False]),
                    "push_notifications": True,
                    "slack_notifications": random.choice([True, False]),
                    "webhook_notifications": True
                }
            }
        }
        
        return tenant_data
    
    @staticmethod
    def _get_company_size(user_count: int) -> str:
        """Determine company size based on user count"""
        if user_count < 10:
            return "startup"
        elif user_count < 50:
            return "small"
        elif user_count < 200:
            return "medium"
        elif user_count < 1000:
            return "large"
        else:
            return "enterprise"
    
    @staticmethod
    def _get_country_for_region(region: str) -> str:
        """Get representative country for region"""
        region_countries = {
            "EU": ["Germany", "France", "Netherlands", "Sweden", "Austria"],
            "US": ["United States"],
            "APAC": ["Japan", "Australia", "Singapore", "South Korea"],
            "LATAM": ["Brazil", "Mexico", "Argentina"],
            "MENA": ["UAE", "Saudi Arabia", "Israel"]
        }
        return random.choice(region_countries.get(region, ["United States"]))
    
    @staticmethod
    def _get_currency_for_region(region: str) -> str:
        """Get primary currency for region"""
        region_currencies = {
            "EU": "EUR",
            "US": "USD", 
            "APAC": "USD",
            "LATAM": "USD",
            "MENA": "USD"
        }
        return region_currencies.get(region, "USD")
    
    @staticmethod
    def _get_timezone_for_region(region: str) -> str:
        """Get primary timezone for region"""
        region_timezones = {
            "EU": "Europe/Berlin",
            "US": "America/New_York",
            "APAC": "Asia/Tokyo",
            "LATAM": "America/Sao_Paulo",
            "MENA": "Asia/Dubai"
        }
        return region_timezones.get(region, "UTC")
    
    @staticmethod
    def _get_locale_for_region(region: str) -> str:
        """Get primary locale for region"""
        region_locales = {
            "EU": "en-GB",
            "US": "en-US",
            "APAC": "en-US",
            "LATAM": "es-ES",
            "MENA": "en-US"
        }
        return region_locales.get(region, "en-US")
    
    @staticmethod
    def _get_date_format_for_region(region: str) -> str:
        """Get date format for region"""
        region_formats = {
            "EU": "DD/MM/YYYY",
            "US": "MM/DD/YYYY",
            "APAC": "YYYY/MM/DD",
            "LATAM": "DD/MM/YYYY",
            "MENA": "DD/MM/YYYY"
        }
        return region_formats.get(region, "MM/DD/YYYY")
    
    @staticmethod
    def _get_currency_format_for_region(region: str) -> str:
        """Get currency format for region"""
        region_formats = {
            "EU": "€#,##0.00",
            "US": "$#,##0.00",
            "APAC": "$#,##0.00",
            "LATAM": "$#,##0.00",
            "MENA": "$#,##0.00"
        }
        return region_formats.get(region, "$#,##0.00")
    
    @staticmethod
    def _get_storage_limit(data_volume: str) -> int:
        """Get storage limit based on data volume"""
        volume_limits = {
            "small": random.randint(10, 50),
            "medium": random.randint(100, 500),
            "large": random.randint(1000, 5000),
            "enterprise": random.randint(10000, 50000)
        }
        return volume_limits.get(data_volume, 100)
    
    @staticmethod
    def _get_api_rate_limit(plan_tier: str) -> int:
        """Get API rate limit based on plan tier"""
        tier_limits = {
            "basic": 1000,
            "professional": 10000,
            "enterprise": 100000
        }
        return tier_limits.get(plan_tier, 1000)
    
    @staticmethod
    def _get_backup_retention(plan_tier: str) -> int:
        """Get backup retention based on plan tier"""
        tier_retention = {
            "basic": 7,
            "professional": 30,
            "enterprise": 90
        }
        return tier_retention.get(plan_tier, 7)
    
    @staticmethod
    def _get_sla_tier(plan_tier: str) -> str:
        """Get SLA tier based on plan"""
        tier_sla = {
            "basic": "99.5%",
            "professional": "99.9%",
            "enterprise": "99.99%"
        }
        return tier_sla.get(plan_tier, "99.5%")


class TenantUserFactory:
    """👥 Factory for generating tenant user test data"""
    
    @staticmethod
    def create_tenant_users(tenant_id: str, count: int, roles: List[str] = None) -> List[Dict[str, Any]]:
        """Generate multiple tenant users with realistic data"""
        if not roles:
            roles = ["admin", "manager", "user", "viewer"]
        
        users = []
        for i in range(count):
            user_role = random.choice(roles)
            
            user_data = {
                "user_id": f"user_{tenant_id}_{i+1}",
                "tenant_id": tenant_id,
                "email": fake.email(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "role": user_role,
                "status": random.choice(["active", "inactive", "pending"]),
                "permissions": TenantUserFactory._get_permissions_for_role(user_role),
                "profile": {
                    "department": random.choice([
                        "Engineering", "Marketing", "Sales", "HR", "Finance", 
                        "Operations", "Customer Success", "Product"
                    ]),
                    "job_title": TenantUserFactory._get_job_title_for_role(user_role),
                    "phone": fake.phone_number(),
                    "timezone": random.choice([
                        "America/New_York", "Europe/London", "Asia/Tokyo", 
                        "Australia/Sydney", "America/Los_Angeles"
                    ]),
                    "language": random.choice(["en", "fr", "de", "es", "it"]),
                    "avatar_url": f"https://api.dicebear.com/7.x/personas/svg?seed={fake.word()}",
                    "bio": fake.text(max_nb_chars=200)
                },
                "authentication": {
                    "password_hash": "$2b$12$" + "x" * 50,  # Mock hash
                    "mfa_enabled": user_role in ["admin", "manager"],
                    "last_login": fake.date_time_between(start_date="-30d", end_date="now"),
                    "login_count": random.randint(1, 500),
                    "failed_login_attempts": random.randint(0, 3),
                    "password_changed_at": fake.date_time_between(start_date="-90d", end_date="now")
                },
                "preferences": {
                    "email_notifications": True,
                    "push_notifications": random.choice([True, False]),
                    "theme": random.choice(["light", "dark", "auto"]),
                    "dashboard_layout": random.choice(["default", "compact", "detailed"]),
                    "data_export_format": random.choice(["json", "csv", "xlsx"])
                },
                "activity": {
                    "created_at": fake.date_time_between(start_date="-1y", end_date="-1d"),
                    "last_activity": fake.date_time_between(start_date="-7d", end_date="now"),
                    "session_count": random.randint(10, 1000),
                    "total_time_spent_minutes": random.randint(1000, 50000)
                }
            }
            users.append(user_data)
        
        return users
    
    @staticmethod
    def _get_permissions_for_role(role: str) -> List[str]:
        """Get permissions based on user role"""
        role_permissions = {
            "admin": [
                "read", "write", "delete", "manage_users", "manage_billing", 
                "manage_security", "export_data", "manage_integrations", "audit_logs"
            ],
            "manager": [
                "read", "write", "manage_users", "export_data", "view_analytics"
            ],
            "user": [
                "read", "write", "export_own_data"
            ],
            "viewer": [
                "read"
            ]
        }
        return role_permissions.get(role, ["read"])
    
    @staticmethod
    def _get_job_title_for_role(role: str) -> str:
        """Get appropriate job title for role"""
        role_titles = {
            "admin": random.choice([
                "System Administrator", "IT Director", "Chief Technology Officer",
                "Platform Administrator", "Infrastructure Manager"
            ]),
            "manager": random.choice([
                "Team Lead", "Project Manager", "Department Manager",
                "Product Manager", "Operations Manager"
            ]),
            "user": random.choice([
                "Software Engineer", "Data Analyst", "Marketing Specialist",
                "Sales Representative", "Customer Success Manager"
            ]),
            "viewer": random.choice([
                "Intern", "Contractor", "External Consultant", "Auditor"
            ])
        }
        return role_titles.get(role, "Team Member")


class TestDataScenarios:
    """🎭 Predefined test scenarios for comprehensive testing"""
    
    @staticmethod
    def get_startup_scenario() -> TenantTestProfile:
        """Small startup company scenario"""
        return TenantTestProfile(
            tenant_type="business",
            plan_tier="basic",
            user_count=random.randint(3, 10),
            data_volume="small",
            compliance_requirements=["GDPR"],
            security_level="standard",
            geographic_region="US",
            industry="technology",
            special_features=["rapid_growth"],
            test_scenario="startup_growth"
        )
    
    @staticmethod
    def get_enterprise_scenario() -> TenantTestProfile:
        """Large enterprise scenario"""
        return TenantTestProfile(
            tenant_type="enterprise",
            plan_tier="enterprise",
            user_count=random.randint(500, 2000),
            data_volume="enterprise",
            compliance_requirements=["GDPR", "SOC2", "HIPAA", "PCI_DSS"],
            security_level="enterprise",
            geographic_region="EU",
            industry="healthcare",
            special_features=["multi_region", "high_availability", "custom_sla"],
            test_scenario="enterprise_complex"
        )
    
    @staticmethod
    def get_healthcare_scenario() -> TenantTestProfile:
        """Healthcare industry scenario"""
        return TenantTestProfile(
            tenant_type="healthcare",
            plan_tier="professional",
            user_count=random.randint(50, 200),
            data_volume="large",
            compliance_requirements=["HIPAA", "GDPR"],
            security_level="high",
            geographic_region="US",
            industry="healthcare",
            special_features=["phi_protection", "audit_trail"],
            test_scenario="healthcare_compliance"
        )
    
    @staticmethod
    def get_financial_scenario() -> TenantTestProfile:
        """Financial services scenario"""
        return TenantTestProfile(
            tenant_type="financial",
            plan_tier="enterprise",
            user_count=random.randint(100, 500),
            data_volume="large",
            compliance_requirements=["PCI_DSS", "SOC2", "GDPR"],
            security_level="enterprise",
            geographic_region="US",
            industry="financial_services",
            special_features=["real_time_monitoring", "fraud_detection"],
            test_scenario="financial_compliance"
        )
    
    @staticmethod
    def get_education_scenario() -> TenantTestProfile:
        """Educational institution scenario"""
        return TenantTestProfile(
            tenant_type="education",
            plan_tier="professional",
            user_count=random.randint(200, 1000),
            data_volume="medium",
            compliance_requirements=["FERPA", "GDPR"],
            security_level="high",
            geographic_region="EU",
            industry="education",
            special_features=["student_data_protection", "parental_consent"],
            test_scenario="education_privacy"
        )
    
    @staticmethod
    def get_government_scenario() -> TenantTestProfile:
        """Government agency scenario"""
        return TenantTestProfile(
            tenant_type="government",
            plan_tier="enterprise",
            user_count=random.randint(300, 800),
            data_volume="large",
            compliance_requirements=["FISMA", "FedRAMP", "GDPR"],
            security_level="government",
            geographic_region="US",
            industry="government",
            special_features=["classified_data", "citizen_privacy", "transparency"],
            test_scenario="government_security"
        )


# Test Environment Fixtures
@pytest.fixture
async def test_tenant_manager():
    """Provide enterprise tenant manager for testing"""
    manager = EnterpriseTenantManager()
    yield manager
    await manager.cleanup()


@pytest.fixture  
async def test_security_manager():
    """Provide security manager for testing"""
    manager = TenantSecurityManager()
    yield manager
    await manager.cleanup()


@pytest.fixture
async def test_isolation_manager():
    """Provide data isolation manager for testing"""
    manager = QuantumDataIsolationManager()
    yield manager
    await manager.cleanup()


@pytest.fixture
def sample_tenant_data():
    """Generate sample tenant data for testing"""
    return TenantDataFactory.create_sample_tenant_data()


@pytest.fixture
def startup_tenant_data():
    """Generate startup scenario tenant data"""
    profile = TestDataScenarios.get_startup_scenario()
    return TenantDataFactory.create_sample_tenant_data(profile)


@pytest.fixture
def enterprise_tenant_data():
    """Generate enterprise scenario tenant data"""
    profile = TestDataScenarios.get_enterprise_scenario()
    return TenantDataFactory.create_sample_tenant_data(profile)


@pytest.fixture
def healthcare_tenant_data():
    """Generate healthcare scenario tenant data"""
    profile = TestDataScenarios.get_healthcare_scenario()
    return TenantDataFactory.create_sample_tenant_data(profile)


@pytest.fixture
def financial_tenant_data():
    """Generate financial scenario tenant data"""
    profile = TestDataScenarios.get_financial_scenario()
    return TenantDataFactory.create_sample_tenant_data(profile)


@pytest.fixture
def sample_tenant_users():
    """Generate sample tenant users"""
    tenant_id = str(uuid.uuid4())
    return TenantUserFactory.create_tenant_users(tenant_id, count=10)


@pytest.fixture
async def test_database():
    """Provide test database connection"""
    # Mock database for testing
    test_db = {
        "tenants": {},
        "users": {},
        "audit_logs": [],
        "security_events": []
    }
    
    yield test_db
    
    # Cleanup
    test_db.clear()


@pytest.fixture
async def mock_external_services():
    """Mock external services for testing"""
    with patch('app.tenancy.services.EmailService') as mock_email, \
         patch('app.tenancy.services.PaymentService') as mock_payment, \
         patch('app.tenancy.services.NotificationService') as mock_notification:
        
        # Configure service mocks
        mock_email.return_value.send_email = AsyncMock(return_value={"status": "sent"})
        mock_payment.return_value.process_payment = AsyncMock(return_value={"status": "success"})
        mock_notification.return_value.send_notification = AsyncMock(return_value={"status": "delivered"})
        
        yield {
            "email": mock_email,
            "payment": mock_payment,
            "notification": mock_notification
        }


@pytest.fixture
def mock_redis():
    """Mock Redis for caching tests"""
    redis_mock = Mock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.expire = AsyncMock(return_value=True)
    
    with patch('aioredis.from_url', return_value=redis_mock):
        yield redis_mock


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests"""
    import time
    import psutil
    
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss
    
    yield {
        "start_time": start_time,
        "start_memory": start_memory,
        "process": process
    }
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance metrics
    print(f"\nTest Performance Metrics:")
    print(f"Execution Time: {execution_time:.3f}s")
    print(f"Memory Delta: {memory_delta / 1024 / 1024:.2f}MB")
    print(f"Peak Memory: {end_memory / 1024 / 1024:.2f}MB")


@pytest.fixture
async def security_test_environment():
    """Setup secure testing environment"""
    test_config = {
        "encryption_key": "test_encryption_key_12345678901234567890123456789012",
        "jwt_secret": "test_jwt_secret_key",
        "api_keys": {
            "test_api_key": "test_tenant_123"
        },
        "security_headers": {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block"
        }
    }
    
    with patch.dict('os.environ', {
        'TEST_MODE': 'true',
        'ENCRYPTION_KEY': test_config["encryption_key"],
        'JWT_SECRET': test_config["jwt_secret"]
    }):
        yield test_config


# Test Data Cleanup Utilities
class TestDataCleaner:
    """🧹 Utility for cleaning up test data"""
    
    @staticmethod
    async def cleanup_test_tenants(tenant_manager, test_prefix: str = "test_"):
        """Clean up test tenants by prefix"""
        tenants = await tenant_manager.list_tenants(
            filter_type="name",
            filter_value=test_prefix,
            limit=1000
        )
        
        for tenant in tenants.get("tenants", []):
            if tenant["name"].startswith(test_prefix):
                await tenant_manager.delete_tenant(tenant["tenant_id"])
    
    @staticmethod
    async def cleanup_test_users(user_manager, test_email_domain: str = "test.example.com"):
        """Clean up test users by email domain"""
        users = await user_manager.list_users(
            filter_type="email",
            filter_value=f"@{test_email_domain}",
            limit=1000
        )
        
        for user in users.get("users", []):
            if user["email"].endswith(test_email_domain):
                await user_manager.delete_user(user["user_id"])


# Enhanced Factory Functions
def create_sample_tenant_data(scenario: str = "default") -> Dict[str, Any]:
    """Create sample tenant data based on scenario"""
    scenario_map = {
        "startup": TestDataScenarios.get_startup_scenario,
        "enterprise": TestDataScenarios.get_enterprise_scenario,
        "healthcare": TestDataScenarios.get_healthcare_scenario,
        "financial": TestDataScenarios.get_financial_scenario,
        "education": TestDataScenarios.get_education_scenario,
        "government": TestDataScenarios.get_government_scenario
    }
    
    if scenario in scenario_map:
        profile = scenario_map[scenario]()
        return TenantDataFactory.create_sample_tenant_data(profile)
    else:
        return TenantDataFactory.create_sample_tenant_data()


def create_sample_user_data(tenant_id: str, role: str = "user") -> Dict[str, Any]:
    """Create sample user data for specific role"""
    users = TenantUserFactory.create_tenant_users(tenant_id, 1, [role])
    return users[0] if users else {}


def create_bulk_test_data(tenant_count: int = 10, users_per_tenant: int = 5) -> Dict[str, Any]:
    """Create bulk test data for load testing"""
    test_data = {
        "tenants": [],
        "users": [],
        "metadata": {
            "tenant_count": tenant_count,
            "total_users": tenant_count * users_per_tenant,
            "generated_at": datetime.utcnow().isoformat()
        }
    }
    
    for i in range(tenant_count):
        # Create tenant with different scenarios
        scenarios = ["startup", "enterprise", "healthcare", "financial"]
        scenario = scenarios[i % len(scenarios)]
        tenant_data = create_sample_tenant_data(scenario)
        tenant_data["name"] = f"Bulk Test Tenant {i+1}"
        
        tenant_id = str(uuid.uuid4())
        tenant_data["tenant_id"] = tenant_id
        test_data["tenants"].append(tenant_data)
        
        # Create users for tenant
        users = TenantUserFactory.create_tenant_users(tenant_id, users_per_tenant)
        test_data["users"].extend(users)
    
    return test_data


# Export main factory functions
__all__ = [
    "TenantDataFactory",
    "TenantUserFactory",
    "TestDataScenarios",
    "TenantTestProfile",
    "TestDataCleaner",
    "create_sample_tenant_data",
    "create_sample_user_data",
    "create_bulk_test_data"
]
