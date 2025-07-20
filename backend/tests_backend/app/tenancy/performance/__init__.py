"""
🧪 Performance Tests - Load, Stress & Benchmark
==============================================

Comprehensive performance testing suite for tenant management including
load testing, stress testing, spike testing, and performance benchmarking.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import psutil
import memory_profiler

from locust import HttpUser, task, between
from hypothesis import given, strategies as st, settings

from app.tenancy import EnterpriseTenantManager
from app.tenancy.models import TenantCreate, TenantUpdate
from app.tenancy.services import TenantLifecycleService, TenantBillingService
from tests_backend.app.tenancy.fixtures.tenant_factories import (
    TenantDataFactory, create_sample_tenant_data
)
from tests_backend.app.tenancy import performance_monitor

pytestmark = pytest.mark.asyncio


class TestTenantCreationPerformance:
    """🏢 Tenant Creation Performance Tests"""
    
    @pytest.fixture
    async def tenant_manager(self):
        """Create EnterpriseTenantManager for performance testing"""
        manager = EnterpriseTenantManager()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.benchmark
    async def test_single_tenant_creation_benchmark(self, benchmark, tenant_manager):
        """Benchmark single tenant creation performance"""
        tenant_data = create_sample_tenant_data()
        
        def create_tenant():
            return asyncio.run(
                tenant_manager.create_enterprise_tenant(TenantCreate(**tenant_data))
            )
        
        result = benchmark(create_tenant)
        
        # Performance assertions
        assert benchmark.stats.mean < 0.1  # Should complete in < 100ms
        assert benchmark.stats.min < 0.05   # Best case < 50ms
        assert benchmark.stats.max < 0.5    # Worst case < 500ms
        
        # Memory usage should be reasonable
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 512  # Less than 512MB
    
    async def test_concurrent_tenant_creation_load(self, tenant_manager):
        """Test concurrent tenant creation under load"""
        concurrent_tenants = 100
        max_execution_time = 30  # seconds
        
        async def create_tenant_with_id(tenant_index: int):
            """Create tenant with unique identifier"""
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Load Test Tenant {tenant_index}"
            tenant_data["slug"] = f"load-test-tenant-{tenant_index}"
            
            start_time = time.perf_counter()
            try:
                result = await tenant_manager.create_enterprise_tenant(
                    TenantCreate(**tenant_data)
                )
                end_time = time.perf_counter()
                
                return {
                    "success": True,
                    "tenant_id": result.tenant_id,
                    "execution_time": end_time - start_time,
                    "index": tenant_index
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "success": False,
                    "error": str(e),
                    "execution_time": end_time - start_time,
                    "index": tenant_index
                }
        
        # Execute concurrent tenant creations
        start_time = time.perf_counter()
        
        tasks = [create_tenant_with_id(i) for i in range(concurrent_tenants)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_execution_time = end_time - start_time
        
        # Analyze results
        successful_creations = [r for r in results if not isinstance(r, Exception) and r.get("success")]
        failed_creations = [r for r in results if isinstance(r, Exception) or not r.get("success")]
        
        execution_times = [r["execution_time"] for r in successful_creations]
        
        # Performance assertions
        assert len(successful_creations) >= concurrent_tenants * 0.95  # 95% success rate
        assert total_execution_time < max_execution_time
        
        # Response time assertions
        if execution_times:
            avg_response_time = statistics.mean(execution_times)
            p95_response_time = statistics.quantiles(execution_times, n=20)[18]  # 95th percentile
            
            assert avg_response_time < 1.0  # Average < 1 second
            assert p95_response_time < 3.0   # 95th percentile < 3 seconds
        
        # Throughput calculation
        throughput = len(successful_creations) / total_execution_time
        assert throughput > 10  # Should handle > 10 tenants per second
        
        # Log performance metrics
        performance_monitor.metrics["concurrent_tenant_creation"] = {
            "total_tenants": concurrent_tenants,
            "successful_creations": len(successful_creations),
            "failed_creations": len(failed_creations),
            "total_execution_time": total_execution_time,
            "average_response_time": statistics.mean(execution_times) if execution_times else 0,
            "throughput_per_second": throughput,
            "success_rate": len(successful_creations) / concurrent_tenants * 100
        }
    
    async def test_tenant_creation_stress_test(self, tenant_manager):
        """Stress test tenant creation beyond normal capacity"""
        stress_multiplier = 5
        base_load = 100
        stress_tenants = base_load * stress_multiplier
        
        # Monitor system resources during stress test
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_cpu = psutil.cpu_percent()
        
        async def stress_create_tenant(tenant_index: int):
            """Create tenant under stress conditions"""
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Stress Test Tenant {tenant_index}"
            
            try:
                result = await tenant_manager.create_enterprise_tenant(
                    TenantCreate(**tenant_data)
                )
                return {"success": True, "tenant_id": result.tenant_id}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Execute stress test
        start_time = time.perf_counter()
        
        # Process in batches to prevent overwhelming system
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, stress_tenants, batch_size):
            batch_end = min(batch_start + batch_size, stress_tenants)
            batch_tasks = [
                stress_create_tenant(i) for i in range(batch_start, batch_end)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze stress test results
        successful_stress = [r for r in all_results if not isinstance(r, Exception) and r.get("success")]
        failed_stress = [r for r in all_results if isinstance(r, Exception) or not r.get("success")]
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        final_cpu = psutil.cpu_percent()
        
        memory_increase = final_memory - initial_memory
        cpu_increase = final_cpu - initial_cpu
        
        # Stress test assertions (more lenient than load test)
        success_rate = len(successful_stress) / stress_tenants * 100
        assert success_rate >= 70  # At least 70% success under stress
        
        # System should remain stable
        assert memory_increase < 1000  # Memory increase < 1GB
        assert final_cpu < 95  # CPU usage should not max out completely
        
        # System should recover
        await asyncio.sleep(5)  # Allow system to stabilize
        recovery_memory = psutil.Process().memory_info().rss / 1024 / 1024
        assert recovery_memory <= final_memory  # Memory should not continue growing
    
    async def test_tenant_creation_spike_test(self, tenant_manager):
        """Test tenant creation during traffic spikes"""
        # Simulate gradual ramp-up followed by sudden spike
        
        # Phase 1: Gradual ramp-up (normal load)
        ramp_up_tenants = 50
        ramp_up_tasks = []
        
        for i in range(ramp_up_tenants):
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Ramp Up Tenant {i}"
            
            task = tenant_manager.create_enterprise_tenant(TenantCreate(**tenant_data))
            ramp_up_tasks.append(task)
            
            # Gradual introduction
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        ramp_up_results = await asyncio.gather(*ramp_up_tasks, return_exceptions=True)
        
        # Phase 2: Sudden spike (high load)
        spike_tenants = 200
        spike_start_time = time.perf_counter()
        
        spike_tasks = []
        for i in range(spike_tenants):
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Spike Tenant {i}"
            
            task = tenant_manager.create_enterprise_tenant(TenantCreate(**tenant_data))
            spike_tasks.append(task)
        
        spike_results = await asyncio.gather(*spike_tasks, return_exceptions=True)
        spike_end_time = time.perf_counter()
        
        spike_duration = spike_end_time - spike_start_time
        
        # Analyze spike performance
        successful_ramp = len([r for r in ramp_up_results if not isinstance(r, Exception)])
        successful_spike = len([r for r in spike_results if not isinstance(r, Exception)])
        
        # Spike test assertions
        ramp_up_success_rate = successful_ramp / ramp_up_tenants * 100
        spike_success_rate = successful_spike / spike_tenants * 100
        
        assert ramp_up_success_rate >= 95  # Ramp-up should be highly successful
        assert spike_success_rate >= 80   # Spike should maintain reasonable success rate
        
        # Spike response time should be reasonable
        spike_throughput = successful_spike / spike_duration
        assert spike_throughput > 5  # Should handle > 5 tenants per second even during spike


class TestTenantQueryPerformance:
    """🔍 Tenant Query Performance Tests"""
    
    @pytest.fixture
    async def tenant_manager_with_data(self):
        """Create tenant manager with pre-populated test data"""
        manager = EnterpriseTenantManager()
        
        # Create test tenants for querying
        test_tenants = []
        for i in range(1000):  # 1000 test tenants
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Query Test Tenant {i}"
            tenant_data["slug"] = f"query-test-tenant-{i}"
            
            try:
                tenant = await manager.create_enterprise_tenant(TenantCreate(**tenant_data))
                test_tenants.append(tenant)
            except Exception:
                pass  # Continue with available tenants
        
        yield manager, test_tenants
        await manager.cleanup()
    
    @pytest.mark.benchmark
    async def test_tenant_lookup_performance(self, benchmark, tenant_manager_with_data):
        """Benchmark tenant lookup performance"""
        manager, test_tenants = tenant_manager_with_data
        
        if not test_tenants:
            pytest.skip("No test tenants available for lookup testing")
        
        # Select random tenant for lookup
        import random
        test_tenant = random.choice(test_tenants)
        
        def lookup_tenant():
            return asyncio.run(manager.get_enterprise_tenant(test_tenant.tenant_id))
        
        result = benchmark(lookup_tenant)
        
        # Lookup should be very fast
        assert benchmark.stats.mean < 0.05  # Average < 50ms
        assert benchmark.stats.max < 0.2    # Max < 200ms
    
    async def test_concurrent_tenant_queries(self, tenant_manager_with_data):
        """Test concurrent tenant query performance"""
        manager, test_tenants = tenant_manager_with_data
        
        if len(test_tenants) < 100:
            pytest.skip("Insufficient test tenants for concurrent query testing")
        
        # Select 100 random tenants for querying
        import random
        query_tenants = random.sample(test_tenants, min(100, len(test_tenants)))
        
        async def query_tenant(tenant):
            """Query single tenant"""
            start_time = time.perf_counter()
            try:
                result = await manager.get_enterprise_tenant(tenant.tenant_id)
                end_time = time.perf_counter()
                return {
                    "success": True,
                    "tenant_id": tenant.tenant_id,
                    "query_time": end_time - start_time
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "success": False,
                    "error": str(e),
                    "query_time": end_time - start_time
                }
        
        # Execute concurrent queries
        start_time = time.perf_counter()
        query_tasks = [query_tenant(tenant) for tenant in query_tenants]
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Analyze query performance
        successful_queries = [r for r in results if not isinstance(r, Exception) and r.get("success")]
        query_times = [r["query_time"] for r in successful_queries]
        
        # Performance assertions
        assert len(successful_queries) >= len(query_tenants) * 0.98  # 98% success rate
        
        if query_times:
            avg_query_time = statistics.mean(query_times)
            p95_query_time = statistics.quantiles(query_times, n=20)[18]
            
            assert avg_query_time < 0.1   # Average query < 100ms
            assert p95_query_time < 0.3   # 95th percentile < 300ms
        
        # Throughput should be high for queries
        throughput = len(successful_queries) / total_time
        assert throughput > 50  # Should handle > 50 queries per second
    
    async def test_tenant_search_performance(self, tenant_manager_with_data):
        """Test tenant search and filtering performance"""
        manager, test_tenants = tenant_manager_with_data
        
        if len(test_tenants) < 100:
            pytest.skip("Insufficient test tenants for search testing")
        
        search_scenarios = [
            {"filter": "plan", "value": "enterprise"},
            {"filter": "status", "value": "active"},
            {"filter": "tier", "value": "platinum"},
            {"filter": "region", "value": "us-east-1"},
            {"filter": "name_contains", "value": "Test"}
        ]
        
        search_results = {}
        
        for scenario in search_scenarios:
            start_time = time.perf_counter()
            
            # Execute search query
            search_result = await manager.search_tenants(
                filter_type=scenario["filter"],
                filter_value=scenario["value"],
                limit=100
            )
            
            end_time = time.perf_counter()
            search_time = end_time - start_time
            
            search_results[scenario["filter"]] = {
                "search_time": search_time,
                "results_count": len(search_result.get("tenants", [])),
                "total_matches": search_result.get("total_count", 0)
            }
            
            # Search should be fast even with large dataset
            assert search_time < 1.0  # Search < 1 second
        
        # All searches should complete quickly
        avg_search_time = statistics.mean([r["search_time"] for r in search_results.values()])
        assert avg_search_time < 0.5  # Average search time < 500ms


class TestTenantResourcePerformance:
    """⚡ Tenant Resource Management Performance Tests"""
    
    async def test_resource_scaling_performance(self):
        """Test performance of tenant resource scaling operations"""
        from app.tenancy.advanced_managers import AdaptiveResourceManager
        
        resource_manager = AdaptiveResourceManager()
        tenant_id = "resource_perf_test"
        
        # Test various scaling scenarios
        scaling_scenarios = [
            {"action": "scale_up", "target_instances": 10, "resource_type": "compute"},
            {"action": "scale_down", "target_instances": 5, "resource_type": "compute"},
            {"action": "scale_up", "target_storage": 1000, "resource_type": "storage"},
            {"action": "scale_up", "target_bandwidth": 10000, "resource_type": "network"}
        ]
        
        scaling_times = []
        
        for scenario in scaling_scenarios:
            start_time = time.perf_counter()
            
            scaling_result = await resource_manager.scale_tenant_resources(
                tenant_id, scenario
            )
            
            end_time = time.perf_counter()
            scaling_time = end_time - start_time
            scaling_times.append(scaling_time)
            
            assert scaling_result.get("status") == "scaling_initiated"
            assert scaling_time < 5.0  # Scaling should initiate quickly
        
        # Average scaling time should be reasonable
        avg_scaling_time = statistics.mean(scaling_times)
        assert avg_scaling_time < 3.0
        
        await resource_manager.cleanup()
    
    async def test_billing_calculation_performance(self):
        """Test performance of billing calculations under load"""
        from app.tenancy.advanced_managers import EnterpriseBillingManager
        
        billing_manager = EnterpriseBillingManager()
        
        # Generate large usage dataset
        usage_data_sets = []
        for tenant_idx in range(100):  # 100 tenants
            usage_data = {
                "tenant_id": f"billing_perf_test_{tenant_idx}",
                "api_calls": 50000 + (tenant_idx * 1000),
                "storage_gb": 1000 + (tenant_idx * 10),
                "bandwidth_gb": 2000 + (tenant_idx * 20),
                "compute_hours": 720,  # Full month
                "premium_features": tenant_idx % 2 == 0  # Every other tenant
            }
            usage_data_sets.append(usage_data)
        
        # Benchmark billing calculations
        start_time = time.perf_counter()
        
        billing_tasks = []
        for usage_data in usage_data_sets:
            task = billing_manager.calculate_usage_bill(
                usage_data["tenant_id"], usage_data, pricing_model={}
            )
            billing_tasks.append(task)
        
        billing_results = await asyncio.gather(*billing_tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_billing_time = end_time - start_time
        
        # Analyze billing performance
        successful_calculations = [r for r in billing_results if not isinstance(r, Exception)]
        
        assert len(successful_calculations) >= len(usage_data_sets) * 0.95  # 95% success
        
        # Billing calculations should be fast
        avg_calculation_time = total_billing_time / len(usage_data_sets)
        assert avg_calculation_time < 0.1  # < 100ms per tenant
        assert total_billing_time < 30  # Total < 30 seconds for 100 tenants
        
        await billing_manager.cleanup()


class TenantLoadTestUser(HttpUser):
    """🔥 Locust Load Test User for Tenant Operations"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Setup for load test user"""
        self.tenant_factory = TenantDataFactory()
        self.created_tenants = []
    
    @task(3)
    def create_tenant(self):
        """Load test task: Create tenant"""
        tenant_data = self.tenant_factory.create_tenant_data()
        
        response = self.client.post(
            "/api/v1/tenants/",
            json=tenant_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            tenant_id = response.json().get("tenant_id")
            if tenant_id:
                self.created_tenants.append(tenant_id)
    
    @task(5)
    def get_tenant(self):
        """Load test task: Get tenant"""
        if self.created_tenants:
            import random
            tenant_id = random.choice(self.created_tenants)
            
            self.client.get(f"/api/v1/tenants/{tenant_id}")
    
    @task(2)
    def update_tenant(self):
        """Load test task: Update tenant"""
        if self.created_tenants:
            import random
            tenant_id = random.choice(self.created_tenants)
            
            update_data = {
                "name": f"Updated Tenant {random.randint(1000, 9999)}",
                "max_users": random.randint(100, 1000)
            }
            
            self.client.patch(
                f"/api/v1/tenants/{tenant_id}",
                json=update_data,
                headers={"Content-Type": "application/json"}
            )
    
    @task(1)
    def search_tenants(self):
        """Load test task: Search tenants"""
        search_params = {
            "plan": "enterprise",
            "status": "active",
            "limit": 50
        }
        
        self.client.get("/api/v1/tenants/search", params=search_params)


class TestMemoryPerformance:
    """🧠 Memory Usage and Performance Tests"""
    
    @memory_profiler.profile
    async def test_tenant_creation_memory_usage(self):
        """Profile memory usage during tenant creation"""
        manager = EnterpriseTenantManager()
        
        # Monitor memory usage while creating tenants
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tenants_created = []
        for i in range(100):  # Create 100 tenants
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Memory Test Tenant {i}"
            
            try:
                tenant = await manager.create_enterprise_tenant(TenantCreate(**tenant_data))
                tenants_created.append(tenant)
                
                # Check for memory leaks every 10 tenants
                if i % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable
                    expected_increase = (i + 1) * 0.5  # ~0.5MB per tenant max
                    assert memory_increase < expected_increase * 2  # Allow 2x buffer
            
            except Exception as e:
                print(f"Failed to create tenant {i}: {e}")
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert total_memory_increase < 100  # Total increase < 100MB
        
        # Cleanup and verify memory release
        await manager.cleanup()
        
        # Allow garbage collection
        import gc
        gc.collect()
        await asyncio.sleep(1)
        
        cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_released = final_memory - cleanup_memory
        
        # Should release significant memory after cleanup
        assert memory_released > total_memory_increase * 0.5  # At least 50% released
    
    async def test_tenant_caching_performance(self):
        """Test tenant caching performance and memory efficiency"""
        manager = EnterpriseTenantManager()
        
        # Create tenants for caching test
        test_tenants = []
        for i in range(50):
            tenant_data = create_sample_tenant_data()
            tenant_data["name"] = f"Cache Test Tenant {i}"
            
            tenant = await manager.create_enterprise_tenant(TenantCreate(**tenant_data))
            test_tenants.append(tenant)
        
        # Test cache performance
        cache_hit_times = []
        cache_miss_times = []
        
        # First access (cache miss)
        for tenant in test_tenants[:10]:
            start_time = time.perf_counter()
            await manager.get_enterprise_tenant(tenant.tenant_id)
            end_time = time.perf_counter()
            cache_miss_times.append(end_time - start_time)
        
        # Second access (cache hit)
        for tenant in test_tenants[:10]:
            start_time = time.perf_counter()
            await manager.get_enterprise_tenant(tenant.tenant_id)
            end_time = time.perf_counter()
            cache_hit_times.append(end_time - start_time)
        
        # Cache hits should be significantly faster
        avg_cache_miss = statistics.mean(cache_miss_times)
        avg_cache_hit = statistics.mean(cache_hit_times)
        
        assert avg_cache_hit < avg_cache_miss * 0.5  # Cache hits 50% faster
        assert avg_cache_hit < 0.01  # Cache hits < 10ms
        
        await manager.cleanup()


# Benchmark configuration
@pytest.fixture(scope="session")
def benchmark_config():
    """Configure benchmarking parameters"""
    return {
        "min_rounds": 5,
        "max_time": 10.0,
        "warmup": True,
        "warmup_iterations": 3
    }


# Performance monitoring fixture
@pytest.fixture(autouse=True)
async def monitor_test_performance(request):
    """Monitor performance for all performance tests"""
    test_name = request.node.name
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    yield
    
    end_time = time.perf_counter()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Store performance metrics
    performance_monitor.metrics[f"perf_{test_name}"] = {
        "execution_time": execution_time,
        "memory_delta": memory_delta,
        "timestamp": datetime.utcnow().isoformat(),
        "test_type": "performance"
    }
    
    # Performance warnings for slow tests
    if execution_time > 60:  # 1 minute
        pytest.warn(f"⚠️ Slow performance test: {test_name} took {execution_time:.2f}s")
    
    if memory_delta > 50:  # 50MB increase
        pytest.warn(f"⚠️ High memory usage: {test_name} used {memory_delta:.2f}MB")
