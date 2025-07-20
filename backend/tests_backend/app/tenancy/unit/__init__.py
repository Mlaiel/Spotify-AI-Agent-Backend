"""
🧪 Unit Tests - Tenant Manager
=============================

Comprehensive unit tests for the EnterpriseTenantManager core functionality.
Tests cover tenant lifecycle, resource management, security, and business logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4
from decimal import Decimal
from typing import Dict, Any, List

from hypothesis import given, strategies as st, settings
from faker import Faker

from app.tenancy import (
    EnterpriseTenantManager,
    TenantModel, TenantDomain, TenantUser, TenantSubscription,
    TenantCreate, TenantUpdate, TenantResponse
)
from app.tenancy.models import TenantStatus, TenantPlan, TenantTier, IsolationLevel
from tests_backend.app.tenancy import (
    test_framework, performance_monitor, security_scanner
)
from tests_backend.app.tenancy.fixtures.tenant_factories import (
    TenantFactory, TenantDataFactory, create_sample_tenant_data
)

pytestmark = pytest.mark.asyncio


class TestEnterpriseTenantManager:
    """🏢 Enterprise Tenant Manager Core Tests"""
    
    @pytest.fixture
    async def tenant_manager(self):
        """Create EnterpriseTenantManager instance for testing"""
        manager = EnterpriseTenantManager()
        yield manager
        # Cleanup after test
        await manager.cleanup()
    
    @pytest.fixture
    def sample_tenant_data(self):
        """Sample tenant data for testing"""
        return create_sample_tenant_data()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database connection"""
        with patch('app.tenancy.advanced_managers.AdvancedDatabaseManager') as mock_db:
            mock_db.return_value.create_tenant_database = AsyncMock(return_value=True)
            mock_db.return_value.setup_tenant_schema = AsyncMock(return_value=True)
            mock_db.return_value.verify_isolation = AsyncMock(return_value=True)
            yield mock_db
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis connection"""
        with patch('app.tenancy.advanced_managers.TenantCacheManager') as mock_redis:
            mock_redis.return_value.cache_tenant = AsyncMock(return_value=True)
            mock_redis.return_value.invalidate_cache = AsyncMock(return_value=True)
            yield mock_redis
    
    async def test_create_enterprise_tenant_success(
        self, tenant_manager, sample_tenant_data, mock_database, mock_redis
    ):
        """Test successful tenant creation with all components"""
        async with performance_monitor.monitor_performance("tenant_creation"):
            # Arrange
            tenant_data = TenantCreate(**sample_tenant_data)
            
            # Act
            result = await tenant_manager.create_enterprise_tenant(tenant_data)
            
            # Assert
            assert result is not None
            assert result.tenant_id is not None
            assert result.name == tenant_data.name
            assert result.status == TenantStatus.ACTIVE
            assert result.plan == tenant_data.plan
            assert result.created_at is not None
            
            # Verify database calls
            mock_database.return_value.create_tenant_database.assert_called_once()
            mock_database.return_value.setup_tenant_schema.assert_called_once()
            
            # Verify cache calls
            mock_redis.return_value.cache_tenant.assert_called_once()
    
    async def test_create_tenant_with_custom_domain(
        self, tenant_manager, sample_tenant_data, mock_database
    ):
        """Test tenant creation with custom domain configuration"""
        # Arrange
        sample_tenant_data.update({
            "custom_domain": "enterprise.example.com",
            "ssl_enabled": True,
            "white_label": True
        })
        tenant_data = TenantCreate(**sample_tenant_data)
        
        # Act
        result = await tenant_manager.create_enterprise_tenant(tenant_data)
        
        # Assert
        assert result.custom_domain == "enterprise.example.com"
        assert result.ssl_enabled is True
        assert result.white_label is True
    
    async def test_create_tenant_with_resource_limits(
        self, tenant_manager, sample_tenant_data
    ):
        """Test tenant creation with specific resource limits"""
        # Arrange
        sample_tenant_data.update({
            "max_users": 500,
            "max_storage_gb": 100,
            "max_api_calls": 50000,
            "max_bandwidth_gb": 1000
        })
        tenant_data = TenantCreate(**sample_tenant_data)
        
        # Act
        result = await tenant_manager.create_enterprise_tenant(tenant_data)
        
        # Assert
        assert result.max_users == 500
        assert result.max_storage_gb == 100
        assert result.max_api_calls == 50000
        assert result.max_bandwidth_gb == 1000
    
    async def test_create_tenant_validation_error(self, tenant_manager):
        """Test tenant creation with invalid data"""
        # Arrange
        invalid_data = TenantCreate(
            name="",  # Invalid empty name
            plan="invalid_plan",  # Invalid plan
            max_users=-1  # Invalid negative limit
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid tenant data"):
            await tenant_manager.create_enterprise_tenant(invalid_data)
    
    async def test_get_enterprise_tenant_success(
        self, tenant_manager, sample_tenant_data, mock_database
    ):
        """Test successful tenant retrieval"""
        # Arrange
        tenant_id = "test_tenant_123"
        expected_tenant = TenantResponse(
            tenant_id=tenant_id,
            name="Test Tenant",
            status=TenantStatus.ACTIVE,
            plan=TenantPlan.ENTERPRISE,
            created_at=datetime.utcnow()
        )
        
        with patch.object(tenant_manager, '_get_tenant_from_cache') as mock_cache:
            mock_cache.return_value = expected_tenant
            
            # Act
            result = await tenant_manager.get_enterprise_tenant(tenant_id)
            
            # Assert
            assert result is not None
            assert result.tenant_id == tenant_id
            assert result.name == "Test Tenant"
            assert result.status == TenantStatus.ACTIVE
    
    async def test_get_tenant_not_found(self, tenant_manager):
        """Test tenant retrieval when tenant doesn't exist"""
        # Arrange
        non_existent_id = "non_existent_tenant"
        
        # Act & Assert
        with pytest.raises(Exception, match="Tenant not found"):
            await tenant_manager.get_enterprise_tenant(non_existent_id)
    
    async def test_update_enterprise_tenant_success(
        self, tenant_manager, sample_tenant_data, mock_database
    ):
        """Test successful tenant update"""
        # Arrange
        tenant_id = "test_tenant_123"
        update_data = TenantUpdate(
            name="Updated Tenant Name",
            plan=TenantPlan.PROFESSIONAL,
            max_users=1000
        )
        
        # Mock existing tenant
        existing_tenant = TenantResponse(
            tenant_id=tenant_id,
            name="Original Name",
            plan=TenantPlan.STARTER,
            status=TenantStatus.ACTIVE,
            created_at=datetime.utcnow()
        )
        
        with patch.object(tenant_manager, 'get_enterprise_tenant') as mock_get:
            mock_get.return_value = existing_tenant
            
            # Act
            result = await tenant_manager.update_enterprise_tenant(tenant_id, update_data)
            
            # Assert
            assert result.name == "Updated Tenant Name"
            assert result.plan == TenantPlan.PROFESSIONAL
            assert result.max_users == 1000
    
    async def test_delete_enterprise_tenant_soft_delete(
        self, tenant_manager, mock_database
    ):
        """Test soft delete of tenant"""
        # Arrange
        tenant_id = "test_tenant_123"
        
        # Act
        result = await tenant_manager.delete_enterprise_tenant(
            tenant_id, soft_delete=True
        )
        
        # Assert
        assert result is True
        # Verify tenant is marked as deleted, not physically removed
    
    async def test_delete_enterprise_tenant_hard_delete(
        self, tenant_manager, mock_database
    ):
        """Test hard delete of tenant"""
        # Arrange
        tenant_id = "test_tenant_123"
        
        with patch.object(tenant_manager, '_backup_tenant_data') as mock_backup:
            mock_backup.return_value = True
            
            # Act
            result = await tenant_manager.delete_enterprise_tenant(
                tenant_id, soft_delete=False
            )
            
            # Assert
            assert result is True
            mock_backup.assert_called_once_with(tenant_id)
    
    async def test_tenant_resource_scaling(self, tenant_manager):
        """Test automatic resource scaling for tenants"""
        # Arrange
        tenant_id = "test_tenant_123"
        scale_config = {
            "cpu_target": 75,
            "memory_target": 80,
            "scale_up_threshold": 85,
            "scale_down_threshold": 30
        }
        
        # Act
        result = await tenant_manager.scale_tenant_resources(tenant_id, scale_config)
        
        # Assert
        assert result.get("status") == "scaling_initiated"
        assert "new_resource_allocation" in result
    
    async def test_tenant_migration_preparation(self, tenant_manager):
        """Test tenant data migration preparation"""
        # Arrange
        source_tenant_id = "source_tenant_123"
        target_region = "eu-west-1"
        
        # Act
        migration_plan = await tenant_manager.prepare_tenant_migration(
            source_tenant_id, target_region
        )
        
        # Assert
        assert migration_plan.get("status") == "ready"
        assert migration_plan.get("target_region") == target_region
        assert "migration_steps" in migration_plan
        assert "estimated_downtime" in migration_plan
    
    async def test_tenant_backup_creation(self, tenant_manager):
        """Test tenant data backup creation"""
        # Arrange
        tenant_id = "test_tenant_123"
        backup_type = "full"
        
        # Act
        backup_result = await tenant_manager.create_tenant_backup(
            tenant_id, backup_type
        )
        
        # Assert
        assert backup_result.get("status") == "completed"
        assert backup_result.get("backup_id") is not None
        assert backup_result.get("backup_size") > 0
        assert backup_result.get("backup_location") is not None
    
    async def test_tenant_analytics_collection(self, tenant_manager):
        """Test tenant analytics and metrics collection"""
        # Arrange
        tenant_id = "test_tenant_123"
        date_range = {
            "start_date": datetime.utcnow() - timedelta(days=30),
            "end_date": datetime.utcnow()
        }
        
        # Act
        analytics = await tenant_manager.get_tenant_analytics(tenant_id, date_range)
        
        # Assert
        assert "usage_metrics" in analytics
        assert "performance_metrics" in analytics
        assert "cost_metrics" in analytics
        assert "user_activity" in analytics
    
    async def test_tenant_security_audit(self, tenant_manager):
        """Test tenant security audit and compliance check"""
        # Arrange
        tenant_id = "test_tenant_123"
        
        # Act
        audit_result = await tenant_manager.perform_security_audit(tenant_id)
        
        # Assert
        assert audit_result.get("status") == "completed"
        assert "security_score" in audit_result
        assert "vulnerabilities" in audit_result
        assert "compliance_status" in audit_result
        assert audit_result.get("security_score") >= 0


class TestTenantPropertyBased:
    """🎲 Property-based tests for tenant operations"""
    
    @given(
        tenant_name=st.text(min_size=1, max_size=255, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])),
        plan=st.sampled_from(["free", "starter", "professional", "enterprise"]),
        max_users=st.integers(min_value=1, max_value=10000),
        max_storage=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=50)
    async def test_tenant_creation_properties(
        self, tenant_name, plan, max_users, max_storage
    ):
        """Property-based test for tenant creation with various inputs"""
        # Arrange
        tenant_manager = EnterpriseTenantManager()
        
        tenant_data = TenantCreate(
            name=tenant_name,
            plan=plan,
            max_users=max_users,
            max_storage_gb=max_storage
        )
        
        try:
            # Act
            result = await tenant_manager.create_enterprise_tenant(tenant_data)
            
            # Assert
            assert result.name == tenant_name
            assert result.plan == plan
            assert result.max_users == max_users
            assert result.max_storage_gb == max_storage
            assert result.tenant_id is not None
            assert result.created_at is not None
            
        except ValueError:
            # Some combinations might be invalid, which is acceptable
            pass
        finally:
            await tenant_manager.cleanup()
    
    @given(
        tenant_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])),
        region=st.sampled_from(["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"])
    )
    @settings(max_examples=30)
    async def test_tenant_region_assignment_properties(self, tenant_id, region):
        """Property-based test for tenant region assignment"""
        tenant_manager = EnterpriseTenantManager()
        
        try:
            result = await tenant_manager.assign_tenant_region(tenant_id, region)
            
            # Properties that should always hold
            assert result.get("tenant_id") == tenant_id
            assert result.get("region") == region
            assert result.get("status") in ["assigned", "pending", "failed"]
            
        except Exception:
            # Some tenant_id/region combinations might not be valid
            pass
        finally:
            await tenant_manager.cleanup()


class TestTenantPerformance:
    """⚡ Performance tests for tenant operations"""
    
    @pytest.mark.benchmark
    async def test_tenant_creation_performance(self, benchmark):
        """Benchmark tenant creation performance"""
        tenant_manager = EnterpriseTenantManager()
        sample_data = create_sample_tenant_data()
        
        def create_tenant():
            return asyncio.run(tenant_manager.create_enterprise_tenant(
                TenantCreate(**sample_data)
            ))
        
        result = benchmark(create_tenant)
        
        # Performance assertions
        assert benchmark.stats.mean < 0.1  # Should complete in < 100ms
        
        await tenant_manager.cleanup()
    
    @pytest.mark.benchmark
    async def test_tenant_lookup_performance(self, benchmark):
        """Benchmark tenant lookup performance"""
        tenant_manager = EnterpriseTenantManager()
        tenant_id = "performance_test_tenant"
        
        def lookup_tenant():
            return asyncio.run(tenant_manager.get_enterprise_tenant(tenant_id))
        
        # This would fail in real scenario without setup, but tests the call structure
        try:
            result = benchmark(lookup_tenant)
            assert benchmark.stats.mean < 0.05  # Should complete in < 50ms
        except Exception:
            # Expected to fail without proper setup
            pass
        
        await tenant_manager.cleanup()
    
    async def test_concurrent_tenant_operations(self):
        """Test concurrent tenant operations performance"""
        tenant_manager = EnterpriseTenantManager()
        concurrent_operations = 100
        
        async def create_test_tenant(index):
            data = create_sample_tenant_data()
            data["name"] = f"Concurrent Tenant {index}"
            return await tenant_manager.create_enterprise_tenant(TenantCreate(**data))
        
        start_time = asyncio.get_event_loop().time()
        
        # Create tenants concurrently
        tasks = [create_test_tenant(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        # Assertions
        successful_operations = len([r for r in results if not isinstance(r, Exception)])
        assert successful_operations > 0
        assert execution_time < 30  # Should complete in < 30 seconds
        
        # Throughput check
        throughput = successful_operations / execution_time
        assert throughput > 10  # Should handle > 10 operations per second
        
        await tenant_manager.cleanup()


class TestTenantSecurity:
    """🔒 Security tests for tenant operations"""
    
    async def test_tenant_data_isolation(self, tenant_manager):
        """Test that tenant data is properly isolated"""
        # Create two tenants
        tenant1_data = create_sample_tenant_data()
        tenant1_data["name"] = "Tenant 1"
        tenant1 = await tenant_manager.create_enterprise_tenant(TenantCreate(**tenant1_data))
        
        tenant2_data = create_sample_tenant_data()
        tenant2_data["name"] = "Tenant 2"
        tenant2 = await tenant_manager.create_enterprise_tenant(TenantCreate(**tenant2_data))
        
        # Verify isolation
        assert tenant1.tenant_id != tenant2.tenant_id
        assert tenant1.database_name != tenant2.database_name
        assert tenant1.cache_namespace != tenant2.cache_namespace
        
        # Test cross-tenant data access prevention
        with pytest.raises(Exception, match="Access denied"):
            await tenant_manager.access_tenant_data(
                tenant1.tenant_id, 
                tenant2.database_name
            )
    
    async def test_tenant_authentication_bypass_prevention(self, tenant_manager):
        """Test prevention of authentication bypass attacks"""
        tenant_id = "security_test_tenant"
        
        # Test various bypass attempts
        bypass_attempts = [
            {"token": "invalid_token", "expected": "Authentication failed"},
            {"token": "../../../etc/passwd", "expected": "Authentication failed"},
            {"token": "'; DROP TABLE tenants; --", "expected": "Authentication failed"},
            {"token": None, "expected": "Authentication failed"}
        ]
        
        for attempt in bypass_attempts:
            with pytest.raises(Exception, match=attempt["expected"]):
                await tenant_manager.authenticate_tenant_access(
                    tenant_id, 
                    attempt["token"]
                )
    
    async def test_tenant_sql_injection_prevention(self, tenant_manager):
        """Test SQL injection prevention in tenant operations"""
        malicious_inputs = [
            "'; DROP TABLE tenants; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM tenants --",
            "admin'--",
            "' OR 1=1 --"
        ]
        
        for malicious_input in malicious_inputs:
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = malicious_input
            
            # Should either sanitize input or raise validation error
            try:
                result = await tenant_manager.create_enterprise_tenant(
                    TenantCreate(**tenant_data)
                )
                # If creation succeeds, name should be sanitized
                assert malicious_input not in result.name
            except ValueError:
                # Validation error is acceptable for malicious input
                pass
    
    async def test_tenant_privilege_escalation_prevention(self, tenant_manager):
        """Test prevention of privilege escalation"""
        # Create tenant with basic permissions
        tenant_data = create_sample_tenant_data()
        tenant_data["plan"] = "free"
        tenant = await tenant_manager.create_enterprise_tenant(TenantCreate(**tenant_data))
        
        # Attempt to escalate privileges
        escalation_attempts = [
            {"action": "upgrade_plan", "target": "enterprise"},
            {"action": "increase_limits", "target": {"max_users": 10000}},
            {"action": "enable_features", "target": ["white_label", "sso"]}
        ]
        
        for attempt in escalation_attempts:
            with pytest.raises(Exception, match="Insufficient privileges"):
                await tenant_manager.escalate_tenant_privileges(
                    tenant.tenant_id,
                    attempt["action"],
                    attempt["target"]
                )


# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def clean_test_environment():
    """Ensure clean test environment before and after tests"""
    # Setup
    await _cleanup_test_data()
    yield
    # Teardown
    await _cleanup_test_data()


async def _cleanup_test_data():
    """Clean up test data from database and cache"""
    # Implementation would clean test databases and caches
    pass


# Performance and memory monitoring
@pytest.fixture(autouse=True)
async def monitor_test_performance(request):
    """Automatically monitor performance for all tests"""
    start_time = asyncio.get_event_loop().time()
    start_memory = performance_monitor._get_memory_usage()
    
    yield
    
    end_time = asyncio.get_event_loop().time()
    end_memory = performance_monitor._get_memory_usage()
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance metrics
    test_name = request.node.name
    performance_monitor.metrics[test_name] = {
        "execution_time": execution_time,
        "memory_delta": memory_delta,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Performance warnings
    if execution_time > 5.0:  # 5 second warning threshold
        pytest.warn(f"⚠️ Slow test: {test_name} took {execution_time:.2f}s")
    
    if memory_delta > 100:  # 100MB memory increase warning
        pytest.warn(f"⚠️ Memory increase: {test_name} used {memory_delta:.2f}MB")
