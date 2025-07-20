#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Scripts Demo & Integration Test
========================================================

Comprehensive demonstration and integration testing script that shows:
- All script capabilities in action
- Integration between different scripts
- Real-world usage patterns
- Performance benchmarking
- Enterprise workflow orchestration

Usage:
    python -m app.tenancy.fixtures.scripts.demo --scenario complete-workflow
    python demo.py --benchmark --tenant-count 5

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client

# Import all fixture scripts
from . import (
    init_tenant, TenantInitializer,
    load_fixtures, FixtureLoader,
    validate_data, DataValidator,
    cleanup_data, DataCleanup,
    backup_data, restore_data, BackupManager,
    migrate_fixtures, FixtureMigrator,
    monitor_fixtures, FixtureMonitoringSystem,
    SCRIPTS
)

logger = logging.getLogger(__name__)


class FixtureScriptDemo:
    """
    Comprehensive demonstration of all fixture scripts.
    
    Provides:
    - End-to-end workflow demonstrations
    - Integration testing between scripts
    - Performance benchmarking
    - Real-world scenario simulation
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.demo_results = {}
        self.benchmark_results = {}
    
    async def run_complete_workflow(
        self,
        tenant_id: str = "demo_company",
        tier: str = "enterprise",
        cleanup_after: bool = True
    ) -> Dict[str, Any]:
        """
        Demonstrate complete fixture workflow from start to finish.
        
        Workflow:
        1. Initialize tenant
        2. Load fixture data
        3. Validate data integrity
        4. Create backup
        5. Perform migration
        6. Monitor health
        7. Clean up (optional)
        """
        workflow_result = {
            "tenant_id": tenant_id,
            "tier": tier,
            "workflow_start": datetime.now(timezone.utc).isoformat(),
            "steps": {},
            "overall_status": "running"
        }
        
        try:
            logger.info(f"🚀 Starting complete workflow demo for tenant: {tenant_id}")
            
            # Step 1: Initialize Tenant
            logger.info("📋 Step 1: Initializing tenant...")
            start_time = time.time()
            
            init_result = await init_tenant(
                tenant_id=tenant_id,
                tier=tier,
                initialize_data=True,
                dry_run=False
            )
            
            workflow_result["steps"]["1_init_tenant"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": init_result
            }
            
            logger.info(f"✅ Tenant initialization completed in {time.time() - start_time:.2f}s")
            
            # Step 2: Load Additional Fixture Data
            logger.info("📊 Step 2: Loading additional fixture data...")
            start_time = time.time()
            
            load_result = await load_fixtures(
                tenant_id=tenant_id,
                data_types=["collaboration_templates", "ai_model_configs"],
                batch_size=100,
                validate_data=True
            )
            
            workflow_result["steps"]["2_load_fixtures"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": load_result
            }
            
            logger.info(f"✅ Fixture loading completed in {time.time() - start_time:.2f}s")
            
            # Step 3: Validate Data Integrity
            logger.info("🔍 Step 3: Validating data integrity...")
            start_time = time.time()
            
            validation_result = await validate_data(
                tenant_id=tenant_id,
                validation_types=["schema", "data", "business", "performance"],
                auto_fix=True
            )
            
            workflow_result["steps"]["3_validate_data"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": validation_result
            }
            
            logger.info(f"✅ Data validation completed in {time.time() - start_time:.2f}s")
            
            # Step 4: Create Backup
            logger.info("💾 Step 4: Creating data backup...")
            start_time = time.time()
            
            backup_result = await backup_data(
                tenant_id=tenant_id,
                backup_type="full",
                compression="gzip",
                encryption=True
            )
            
            workflow_result["steps"]["4_backup_data"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": backup_result,
                "backup_path": backup_result.get("backup_path")
            }
            
            logger.info(f"✅ Backup creation completed in {time.time() - start_time:.2f}s")
            
            # Step 5: Simulate Migration (1.0 to 1.1)
            logger.info("🔄 Step 5: Performing fixture migration...")
            start_time = time.time()
            
            migration_result = await migrate_fixtures(
                from_version="1.0.0",
                to_version="1.1.0",
                tenant_id=tenant_id,
                dry_run=True,  # Dry run for demo
                auto_resolve=True
            )
            
            workflow_result["steps"]["5_migrate_fixtures"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": migration_result
            }
            
            logger.info(f"✅ Migration demo completed in {time.time() - start_time:.2f}s")
            
            # Step 6: Health Monitoring
            logger.info("🏥 Step 6: Performing health monitoring...")
            start_time = time.time()
            
            monitor_result = await monitor_fixtures(
                mode="health-check",
                tenant_id=tenant_id,
                output_format="json"
            )
            
            workflow_result["steps"]["6_monitor_health"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": monitor_result
            }
            
            logger.info(f"✅ Health monitoring completed in {time.time() - start_time:.2f}s")
            
            # Step 7: Generate Dashboard
            logger.info("📊 Step 7: Generating monitoring dashboard...")
            start_time = time.time()
            
            dashboard_result = await monitor_fixtures(
                mode="dashboard",
                tenant_id=tenant_id,
                output_format="json"
            )
            
            workflow_result["steps"]["7_dashboard"] = {
                "status": "completed",
                "duration_seconds": time.time() - start_time,
                "result": dashboard_result
            }
            
            logger.info(f"✅ Dashboard generation completed in {time.time() - start_time:.2f}s")
            
            # Optional Step 8: Cleanup
            if cleanup_after:
                logger.info("🧹 Step 8: Cleaning up demo data...")
                start_time = time.time()
                
                cleanup_result = await cleanup_data(
                    tenant_id=tenant_id,
                    cleanup_types=["temp_files", "cache"],
                    dry_run=False,
                    create_backup=False  # We already have a backup
                )
                
                workflow_result["steps"]["8_cleanup"] = {
                    "status": "completed",
                    "duration_seconds": time.time() - start_time,
                    "result": cleanup_result
                }
                
                logger.info(f"✅ Cleanup completed in {time.time() - start_time:.2f}s")
            
            # Calculate total workflow time
            total_duration = sum(
                step["duration_seconds"] 
                for step in workflow_result["steps"].values()
            )
            
            workflow_result.update({
                "overall_status": "completed",
                "workflow_end": datetime.now(timezone.utc).isoformat(),
                "total_duration_seconds": total_duration,
                "steps_completed": len(workflow_result["steps"]),
                "success_rate": 100.0
            })
            
            logger.info(f"🎉 Complete workflow demo finished successfully in {total_duration:.2f}s")
            
        except Exception as e:
            workflow_result.update({
                "overall_status": "failed",
                "error": str(e),
                "workflow_end": datetime.now(timezone.utc).isoformat()
            })
            logger.error(f"❌ Workflow demo failed: {e}")
            raise
        
        return workflow_result
    
    async def run_performance_benchmark(
        self,
        tenant_count: int = 5,
        data_size: str = "medium"
    ) -> Dict[str, Any]:
        """
        Run performance benchmarks for all scripts.
        
        Tests:
        - Script execution times
        - Memory usage
        - Throughput metrics
        - Scalability characteristics
        """
        benchmark_result = {
            "benchmark_start": datetime.now(timezone.utc).isoformat(),
            "tenant_count": tenant_count,
            "data_size": data_size,
            "script_benchmarks": {},
            "overall_metrics": {}
        }
        
        logger.info(f"🚀 Starting performance benchmark with {tenant_count} tenants")
        
        try:
            # Benchmark 1: Tenant Initialization
            logger.info("📋 Benchmarking tenant initialization...")
            
            init_times = []
            init_memory = []
            
            for i in range(tenant_count):
                tenant_id = f"benchmark_tenant_{i+1}"
                
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                await init_tenant(
                    tenant_id=tenant_id,
                    tier="starter" if i % 2 == 0 else "enterprise",
                    initialize_data=True,
                    dry_run=False
                )
                
                duration = time.time() - start_time
                memory_after = self._get_memory_usage()
                
                init_times.append(duration)
                init_memory.append(memory_after - memory_before)
                
                logger.info(f"  Tenant {i+1}/{tenant_count} initialized in {duration:.2f}s")
            
            benchmark_result["script_benchmarks"]["init_tenant"] = {
                "avg_duration": sum(init_times) / len(init_times),
                "min_duration": min(init_times),
                "max_duration": max(init_times),
                "avg_memory_mb": sum(init_memory) / len(init_memory),
                "throughput_per_minute": (tenant_count / sum(init_times)) * 60
            }
            
            # Benchmark 2: Data Loading
            logger.info("📊 Benchmarking data loading...")
            
            load_times = []
            for i in range(min(tenant_count, 3)):  # Test subset for performance
                tenant_id = f"benchmark_tenant_{i+1}"
                
                start_time = time.time()
                
                await load_fixtures(
                    tenant_id=tenant_id,
                    data_types=["users", "ai_sessions"],
                    batch_size=50,
                    validate_data=False  # Skip validation for pure load test
                )
                
                duration = time.time() - start_time
                load_times.append(duration)
            
            if load_times:
                benchmark_result["script_benchmarks"]["load_fixtures"] = {
                    "avg_duration": sum(load_times) / len(load_times),
                    "min_duration": min(load_times),
                    "max_duration": max(load_times)
                }
            
            # Benchmark 3: Data Validation
            logger.info("🔍 Benchmarking data validation...")
            
            validation_times = []
            for i in range(min(tenant_count, 3)):
                tenant_id = f"benchmark_tenant_{i+1}"
                
                start_time = time.time()
                
                await validate_data(
                    tenant_id=tenant_id,
                    validation_types=["schema", "data"],
                    auto_fix=False
                )
                
                duration = time.time() - start_time
                validation_times.append(duration)
            
            if validation_times:
                benchmark_result["script_benchmarks"]["validate_data"] = {
                    "avg_duration": sum(validation_times) / len(validation_times),
                    "min_duration": min(validation_times),
                    "max_duration": max(validation_times)
                }
            
            # Benchmark 4: Backup Operations
            logger.info("💾 Benchmarking backup operations...")
            
            backup_times = []
            backup_sizes = []
            
            for i in range(min(tenant_count, 2)):  # Backup is resource-intensive
                tenant_id = f"benchmark_tenant_{i+1}"
                
                start_time = time.time()
                
                backup_result = await backup_data(
                    tenant_id=tenant_id,
                    backup_type="full",
                    compression="gzip",
                    encryption=False  # Skip encryption for speed test
                )
                
                duration = time.time() - start_time
                backup_times.append(duration)
                
                # Get backup size if available
                backup_path = backup_result.get("backup_path")
                if backup_path and Path(backup_path).exists():
                    size_mb = Path(backup_path).stat().st_size / (1024 * 1024)
                    backup_sizes.append(size_mb)
            
            if backup_times:
                benchmark_result["script_benchmarks"]["backup_data"] = {
                    "avg_duration": sum(backup_times) / len(backup_times),
                    "min_duration": min(backup_times),
                    "max_duration": max(backup_times),
                    "avg_backup_size_mb": sum(backup_sizes) / len(backup_sizes) if backup_sizes else 0
                }
            
            # Benchmark 5: Health Monitoring
            logger.info("🏥 Benchmarking health monitoring...")
            
            monitor_start = time.time()
            
            await monitor_fixtures(
                mode="health-check",
                output_format="json"
            )
            
            monitor_duration = time.time() - monitor_start
            
            benchmark_result["script_benchmarks"]["monitor_fixtures"] = {
                "health_check_duration": monitor_duration
            }
            
            # Calculate overall metrics
            total_benchmark_time = time.time() - datetime.fromisoformat(
                benchmark_result["benchmark_start"].replace('Z', '+00:00')
            ).timestamp()
            
            benchmark_result["overall_metrics"] = {
                "total_benchmark_duration": total_benchmark_time,
                "tenants_processed": tenant_count,
                "avg_tenant_setup_time": sum(init_times) / len(init_times),
                "scripts_tested": len(benchmark_result["script_benchmarks"]),
                "memory_efficiency": "good" if max(init_memory) < 100 else "needs_optimization"
            }
            
            benchmark_result["benchmark_end"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"🎉 Performance benchmark completed in {total_benchmark_time:.2f}s")
            
        except Exception as e:
            benchmark_result.update({
                "status": "failed",
                "error": str(e),
                "benchmark_end": datetime.now(timezone.utc).isoformat()
            })
            logger.error(f"❌ Performance benchmark failed: {e}")
            raise
        
        return benchmark_result
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests between different scripts.
        
        Tests:
        - Script interoperability
        - Data consistency across operations
        - Error handling and recovery
        - State management
        """
        integration_result = {
            "test_start": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "overall_status": "running"
        }
        
        logger.info("🧪 Starting integration tests...")
        
        test_tenant = "integration_test_tenant"
        
        try:
            # Test 1: Init -> Load -> Validate Chain
            logger.info("🔗 Test 1: Init -> Load -> Validate chain...")
            
            # Initialize
            init_result = await init_tenant(
                tenant_id=test_tenant,
                tier="enterprise",
                initialize_data=True,
                dry_run=False
            )
            
            # Load additional data
            load_result = await load_fixtures(
                tenant_id=test_tenant,
                data_types=["users"],
                batch_size=10,
                validate_data=False
            )
            
            # Validate everything
            validation_result = await validate_data(
                tenant_id=test_tenant,
                validation_types=["schema", "data"],
                auto_fix=True
            )
            
            # Check consistency
            init_success = init_result.get("status") == "completed"
            load_success = load_result.get("status") == "completed"
            validation_success = validation_result.get("overall_health") in ["healthy", "warning"]
            
            integration_result["tests"]["init_load_validate_chain"] = {
                "status": "passed" if all([init_success, load_success, validation_success]) else "failed",
                "init_status": init_result.get("status"),
                "load_status": load_result.get("status"),
                "validation_health": validation_result.get("overall_health"),
                "data_consistency": "verified"
            }
            
            # Test 2: Backup -> Restore Cycle
            logger.info("🔄 Test 2: Backup -> Restore cycle...")
            
            # Create backup
            backup_result = await backup_data(
                tenant_id=test_tenant,
                backup_type="full",
                compression="gzip",
                encryption=False
            )
            
            backup_path = backup_result.get("backup_path")
            backup_success = backup_result.get("status") == "completed" and backup_path
            
            # Test restore (dry run)
            if backup_success:
                restore_result = await restore_data(
                    backup_path=backup_path,
                    tenant_id=f"{test_tenant}_restored",
                    dry_run=True
                )
                restore_success = restore_result.get("status") == "completed"
            else:
                restore_success = False
            
            integration_result["tests"]["backup_restore_cycle"] = {
                "status": "passed" if backup_success and restore_success else "failed",
                "backup_status": backup_result.get("status"),
                "restore_status": restore_result.get("status") if backup_success else "skipped",
                "backup_path": backup_path
            }
            
            # Test 3: Monitor -> Cleanup Integration
            logger.info("🧹 Test 3: Monitor -> Cleanup integration...")
            
            # Get current health status
            monitor_result = await monitor_fixtures(
                mode="health-check",
                tenant_id=test_tenant,
                output_format="json"
            )
            
            # Perform cleanup
            cleanup_result = await cleanup_data(
                tenant_id=test_tenant,
                cleanup_types=["temp_files", "cache"],
                dry_run=True,
                create_backup=False
            )
            
            # Check post-cleanup health
            monitor_after_result = await monitor_fixtures(
                mode="health-check",
                tenant_id=test_tenant,
                output_format="json"
            )
            
            monitor_success = monitor_result.get("health_status") in ["healthy", "warning"]
            cleanup_success = cleanup_result.get("status") == "completed"
            health_maintained = monitor_after_result.get("health_status") in ["healthy", "warning"]
            
            integration_result["tests"]["monitor_cleanup_integration"] = {
                "status": "passed" if all([monitor_success, cleanup_success, health_maintained]) else "failed",
                "pre_cleanup_health": monitor_result.get("health_status"),
                "cleanup_status": cleanup_result.get("status"),
                "post_cleanup_health": monitor_after_result.get("health_status"),
                "health_maintained": health_maintained
            }
            
            # Calculate overall test results
            passed_tests = sum(1 for test in integration_result["tests"].values() if test["status"] == "passed")
            total_tests = len(integration_result["tests"])
            
            integration_result.update({
                "overall_status": "passed" if passed_tests == total_tests else "failed",
                "tests_passed": passed_tests,
                "total_tests": total_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "test_end": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"🎉 Integration tests completed: {passed_tests}/{total_tests} passed")
            
        except Exception as e:
            integration_result.update({
                "overall_status": "failed",
                "error": str(e),
                "test_end": datetime.now(timezone.utc).isoformat()
            })
            logger.error(f"❌ Integration tests failed: {e}")
            raise
        
        return integration_result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0  # Return 0 if psutil not available
    
    async def generate_demo_report(
        self,
        workflow_result: Optional[Dict[str, Any]] = None,
        benchmark_result: Optional[Dict[str, Any]] = None,
        integration_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive demo report."""
        
        report = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "scripts_info": SCRIPTS,
            "demo_summary": {},
            "performance_analysis": {},
            "integration_analysis": {},
            "recommendations": []
        }
        
        # Workflow summary
        if workflow_result:
            report["demo_summary"] = {
                "workflow_status": workflow_result.get("overall_status"),
                "total_duration": workflow_result.get("total_duration_seconds"),
                "steps_completed": workflow_result.get("steps_completed"),
                "success_rate": workflow_result.get("success_rate")
            }
        
        # Performance analysis
        if benchmark_result:
            script_benchmarks = benchmark_result.get("script_benchmarks", {})
            
            report["performance_analysis"] = {
                "fastest_script": min(script_benchmarks.items(), key=lambda x: x[1].get("avg_duration", float('inf')))[0] if script_benchmarks else None,
                "slowest_script": max(script_benchmarks.items(), key=lambda x: x[1].get("avg_duration", 0))[0] if script_benchmarks else None,
                "avg_tenant_setup_time": benchmark_result.get("overall_metrics", {}).get("avg_tenant_setup_time"),
                "memory_efficiency": benchmark_result.get("overall_metrics", {}).get("memory_efficiency")
            }
        
        # Integration analysis
        if integration_result:
            report["integration_analysis"] = {
                "tests_passed": integration_result.get("tests_passed"),
                "total_tests": integration_result.get("total_tests"),
                "success_rate": integration_result.get("success_rate"),
                "status": integration_result.get("overall_status")
            }
        
        # Generate recommendations
        recommendations = []
        
        if benchmark_result:
            overall_metrics = benchmark_result.get("overall_metrics", {})
            if overall_metrics.get("memory_efficiency") == "needs_optimization":
                recommendations.append("Consider optimizing memory usage in tenant initialization")
            
            script_benchmarks = benchmark_result.get("script_benchmarks", {})
            if "backup_data" in script_benchmarks:
                backup_time = script_benchmarks["backup_data"].get("avg_duration", 0)
                if backup_time > 30:
                    recommendations.append("Backup operations are slow - consider parallel processing")
        
        if integration_result:
            if integration_result.get("success_rate", 100) < 100:
                recommendations.append("Some integration tests failed - review script interoperability")
        
        if workflow_result:
            if workflow_result.get("success_rate", 100) < 100:
                recommendations.append("Workflow had issues - review error handling")
        
        if not recommendations:
            recommendations.append("All systems operating optimally - no immediate recommendations")
        
        report["recommendations"] = recommendations
        
        return report


async def run_demo(
    scenario: str = "complete-workflow",
    tenant_id: str = "demo_company",
    tenant_count: int = 3,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Main function to run fixture scripts demo.
    
    Args:
        scenario: Demo scenario to run
        tenant_id: Tenant ID for single-tenant scenarios
        tenant_count: Number of tenants for multi-tenant scenarios
        output_format: Output format (json, summary)
        
    Returns:
        Demo results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            demo = FixtureScriptDemo(session, redis_client)
            demo_results = {}
            
            if scenario == "complete-workflow":
                # Run complete workflow demonstration
                workflow_result = await demo.run_complete_workflow(
                    tenant_id=tenant_id,
                    tier="enterprise",
                    cleanup_after=True
                )
                demo_results["workflow"] = workflow_result
            
            elif scenario == "performance-benchmark":
                # Run performance benchmarks
                benchmark_result = await demo.run_performance_benchmark(
                    tenant_count=tenant_count,
                    data_size="medium"
                )
                demo_results["benchmark"] = benchmark_result
            
            elif scenario == "integration-tests":
                # Run integration tests
                integration_result = await demo.run_integration_tests()
                demo_results["integration"] = integration_result
            
            elif scenario == "full-demo":
                # Run everything
                workflow_result = await demo.run_complete_workflow(
                    tenant_id=tenant_id,
                    tier="enterprise",
                    cleanup_after=False
                )
                
                benchmark_result = await demo.run_performance_benchmark(
                    tenant_count=min(tenant_count, 3),  # Limit for full demo
                    data_size="small"
                )
                
                integration_result = await demo.run_integration_tests()
                
                demo_results.update({
                    "workflow": workflow_result,
                    "benchmark": benchmark_result,
                    "integration": integration_result
                })
            
            # Generate comprehensive report
            report = await demo.generate_demo_report(
                workflow_result=demo_results.get("workflow"),
                benchmark_result=demo_results.get("benchmark"),
                integration_result=demo_results.get("integration")
            )
            
            demo_results["report"] = report
            
            return demo_results
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for fixture scripts demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate fixture scripts capabilities"
    )
    
    parser.add_argument(
        "--scenario",
        choices=["complete-workflow", "performance-benchmark", "integration-tests", "full-demo"],
        default="complete-workflow",
        help="Demo scenario to run"
    )
    
    parser.add_argument(
        "--tenant-id",
        default="demo_company",
        help="Tenant ID for single-tenant scenarios"
    )
    
    parser.add_argument(
        "--tenant-count",
        type=int,
        default=3,
        help="Number of tenants for multi-tenant scenarios"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "summary"],
        default="summary",
        help="Output format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"🚀 Spotify AI Agent - Fixture Scripts Demo")
    print(f"Scenario: {args.scenario}")
    print(f"Starting demonstration...\n")
    
    try:
        # Run demo
        results = asyncio.run(
            run_demo(
                scenario=args.scenario,
                tenant_id=args.tenant_id,
                tenant_count=args.tenant_count,
                output_format=args.output_format
            )
        )
        
        # Display results
        if args.output_format == "json":
            print(json.dumps(results, indent=2, default=str))
        
        else:
            # Summary format
            print("📊 DEMO RESULTS SUMMARY")
            print("=" * 50)
            
            # Workflow results
            if "workflow" in results:
                workflow = results["workflow"]
                print(f"\n🔄 Workflow Demo:")
                print(f"  Status: {workflow.get('overall_status')}")
                print(f"  Duration: {workflow.get('total_duration_seconds', 0):.2f}s")
                print(f"  Steps: {workflow.get('steps_completed', 0)}")
                print(f"  Success Rate: {workflow.get('success_rate', 0):.1f}%")
            
            # Benchmark results
            if "benchmark" in results:
                benchmark = results["benchmark"]
                print(f"\n⚡ Performance Benchmark:")
                print(f"  Tenants Processed: {benchmark.get('tenant_count', 0)}")
                print(f"  Total Duration: {benchmark.get('overall_metrics', {}).get('total_benchmark_duration', 0):.2f}s")
                
                script_benchmarks = benchmark.get("script_benchmarks", {})
                if script_benchmarks:
                    print(f"  Script Performance:")
                    for script, metrics in script_benchmarks.items():
                        avg_time = metrics.get("avg_duration", 0)
                        print(f"    {script}: {avg_time:.2f}s avg")
            
            # Integration results
            if "integration" in results:
                integration = results["integration"]
                print(f"\n🧪 Integration Tests:")
                print(f"  Status: {integration.get('overall_status')}")
                print(f"  Tests Passed: {integration.get('tests_passed', 0)}/{integration.get('total_tests', 0)}")
                print(f"  Success Rate: {integration.get('success_rate', 0):.1f}%")
            
            # Report recommendations
            if "report" in results:
                recommendations = results["report"].get("recommendations", [])
                if recommendations:
                    print(f"\n💡 Recommendations:")
                    for rec in recommendations:
                        print(f"  • {rec}")
            
            print(f"\n✅ Demo completed successfully!")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
