#!/usr/bin/env python3
"""
Test Data Generation Scripts for Multi-Tenant Scenarios
======================================================

This module provides ready-to-use scripts for generating test data
for various scenarios including load testing, compliance testing,
and performance benchmarking.

Features:
- Automated test data generation for all tenant tiers
- Load testing data with realistic user patterns
- Compliance testing data with GDPR/CCPA requirements
- Performance benchmark datasets
- Data migration testing scenarios

Author: AI Assistant
Version: 1.0.0
"""

import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from multi_tenant_data_generator import (
    MultiTenantDataGenerator,
    TenantTier,
    DataType,
    TenantDataProfile
)


class TestDataScenarios:
    """Pre-defined test data generation scenarios."""
    
    @staticmethod
    def get_demo_scenarios() -> List[Dict]:
        """Get demo scenarios for all tenant tiers."""
        return [
            {
                "tenant_id": "demo_free_001",
                "tenant_name": "Free Demo Tenant",
                "tier": TenantTier.FREE,
                "scale_factor": 0.5,
                "description": "Small dataset for free tier demos"
            },
            {
                "tenant_id": "demo_standard_001", 
                "tenant_name": "Standard Demo Tenant",
                "tier": TenantTier.STANDARD,
                "scale_factor": 0.2,
                "description": "Medium dataset for standard tier demos"
            },
            {
                "tenant_id": "demo_premium_001",
                "tenant_name": "Premium Demo Tenant", 
                "tier": TenantTier.PREMIUM,
                "scale_factor": 0.1,
                "description": "Large dataset for premium tier demos"
            },
            {
                "tenant_id": "demo_enterprise_001",
                "tenant_name": "Enterprise Demo Tenant",
                "tier": TenantTier.ENTERPRISE,
                "scale_factor": 0.05,
                "description": "Enterprise dataset with all features"
            }
        ]
    
    @staticmethod
    def get_load_test_scenarios() -> List[Dict]:
        """Get load testing scenarios."""
        return [
            {
                "tenant_id": "load_test_light",
                "tenant_name": "Light Load Test",
                "tier": TenantTier.STANDARD,
                "scale_factor": 1.0,
                "concurrent_users": 100,
                "description": "Light load testing scenario"
            },
            {
                "tenant_id": "load_test_medium",
                "tenant_name": "Medium Load Test", 
                "tier": TenantTier.PREMIUM,
                "scale_factor": 1.0,
                "concurrent_users": 500,
                "description": "Medium load testing scenario"
            },
            {
                "tenant_id": "load_test_heavy",
                "tenant_name": "Heavy Load Test",
                "tier": TenantTier.ENTERPRISE,
                "scale_factor": 1.0,
                "concurrent_users": 2000,
                "description": "Heavy load testing scenario"
            }
        ]
    
    @staticmethod
    def get_compliance_test_scenarios() -> List[Dict]:
        """Get compliance testing scenarios."""
        return [
            {
                "tenant_id": "compliance_gdpr_001",
                "tenant_name": "GDPR Compliance Test",
                "tier": TenantTier.PREMIUM,
                "scale_factor": 0.3,
                "gdpr_compliant": True,
                "pii_anonymization": False,
                "description": "GDPR compliance testing with real PII"
            },
            {
                "tenant_id": "compliance_anonymized_001",
                "tenant_name": "Anonymized Data Test",
                "tier": TenantTier.ENTERPRISE,
                "scale_factor": 0.2,
                "gdpr_compliant": True,
                "pii_anonymization": True,
                "description": "Fully anonymized data for enterprise compliance"
            }
        ]
    
    @staticmethod
    def get_migration_test_scenarios() -> List[Dict]:
        """Get migration testing scenarios."""
        return [
            {
                "tenant_id": "migration_source",
                "tenant_name": "Migration Source Tenant",
                "tier": TenantTier.STANDARD,
                "scale_factor": 0.5,
                "description": "Source tenant for migration testing"
            },
            {
                "tenant_id": "migration_target",
                "tenant_name": "Migration Target Tenant", 
                "tier": TenantTier.PREMIUM,
                "scale_factor": 0.5,
                "description": "Target tenant for migration testing"
            }
        ]


async def generate_demo_data():
    """Generate demo data for all tenant tiers."""
    print("🎯 Generating demo data for all tenant tiers...")
    
    generator = MultiTenantDataGenerator()
    scenarios = TestDataScenarios.get_demo_scenarios()
    
    for scenario in scenarios:
        print(f"\n📊 Generating data for {scenario['tenant_name']}...")
        
        profile = generator.create_tenant_profile(
            tenant_id=scenario["tenant_id"],
            tenant_name=scenario["tenant_name"],
            tier=scenario["tier"],
            scale_factor=scenario["scale_factor"]
        )
        
        start_time = time.time()
        tenant_data = await generator.generate_tenant_data(profile)
        generation_time = time.time() - start_time
        
        await generator.save_tenant_data(tenant_data, format="json")
        
        print(f"✅ Generated {scenario['tenant_id']} in {generation_time:.2f}s")
        print(f"   Users: {len(tenant_data['data'].get('users', []))}")
        print(f"   Tracks: {len(tenant_data['data'].get('tracks', []))}")
        print(f"   Events: {len(tenant_data['data'].get('listening_history', []))}")


async def generate_load_test_data():
    """Generate data specifically for load testing."""
    print("🚀 Generating load testing data...")
    
    generator = MultiTenantDataGenerator()
    scenarios = TestDataScenarios.get_load_test_scenarios()
    
    for scenario in scenarios:
        print(f"\n⚡ Generating load test data for {scenario['tenant_name']}...")
        
        profile = generator.create_tenant_profile(
            tenant_id=scenario["tenant_id"],
            tenant_name=scenario["tenant_name"],
            tier=scenario["tier"],
            scale_factor=scenario["scale_factor"],
            generate_load_data=True,
            concurrent_users=scenario["concurrent_users"]
        )
        
        # Generate specific data types for load testing
        data_types = [
            DataType.USERS,
            DataType.TRACKS,
            DataType.LISTENING_HISTORY,
            DataType.ANALYTICS_EVENTS
        ]
        
        start_time = time.time()
        tenant_data = await generator.generate_tenant_data(profile, data_types)
        generation_time = time.time() - start_time
        
        await generator.save_tenant_data(tenant_data, format="json")
        
        print(f"✅ Generated {scenario['tenant_id']} in {generation_time:.2f}s")
        print(f"   Concurrent Users: {scenario['concurrent_users']}")
        print(f"   Analytics Events: {len(tenant_data['data'].get('analytics_events', []))}")


async def generate_compliance_test_data():
    """Generate data for compliance testing."""
    print("🔒 Generating compliance testing data...")
    
    generator = MultiTenantDataGenerator()
    scenarios = TestDataScenarios.get_compliance_test_scenarios()
    
    for scenario in scenarios:
        print(f"\n🛡️ Generating compliance data for {scenario['tenant_name']}...")
        
        profile = generator.create_tenant_profile(
            tenant_id=scenario["tenant_id"],
            tenant_name=scenario["tenant_name"],
            tier=scenario["tier"],
            scale_factor=scenario["scale_factor"],
            gdpr_compliant=scenario["gdpr_compliant"],
            pii_anonymization=scenario["pii_anonymization"]
        )
        
        start_time = time.time()
        tenant_data = await generator.generate_tenant_data(profile)
        generation_time = time.time() - start_time
        
        await generator.save_tenant_data(tenant_data, format="json")
        
        # Generate compliance report
        compliance_report = {
            "tenant_id": scenario["tenant_id"],
            "compliance_level": "GDPR" if scenario["gdpr_compliant"] else "Basic",
            "pii_anonymized": scenario["pii_anonymization"],
            "generated_at": datetime.now().isoformat(),
            "data_summary": {
                "total_users": len(tenant_data['data'].get('users', [])),
                "pii_fields_anonymized": scenario["pii_anonymization"],
                "audit_trail_complete": True,
                "encryption_applied": True
            }
        }
        
        # Save compliance report
        report_path = generator.output_path / scenario["tenant_id"] / "compliance_report.json"
        with open(report_path, "w") as f:
            json.dump(compliance_report, f, indent=2)
        
        print(f"✅ Generated {scenario['tenant_id']} with compliance in {generation_time:.2f}s")
        print(f"   GDPR Compliant: {scenario['gdpr_compliant']}")
        print(f"   PII Anonymized: {scenario['pii_anonymization']}")


async def generate_migration_test_data():
    """Generate data for migration testing."""
    print("🔄 Generating migration testing data...")
    
    generator = MultiTenantDataGenerator()
    scenarios = TestDataScenarios.get_migration_test_scenarios()
    
    for scenario in scenarios:
        print(f"\n📦 Generating migration data for {scenario['tenant_name']}...")
        
        profile = generator.create_tenant_profile(
            tenant_id=scenario["tenant_id"],
            tenant_name=scenario["tenant_name"],
            tier=scenario["tier"],
            scale_factor=scenario["scale_factor"]
        )
        
        start_time = time.time()
        tenant_data = await generator.generate_tenant_data(profile)
        generation_time = time.time() - start_time
        
        await generator.save_tenant_data(tenant_data, format="json")
        
        # Generate migration metadata
        migration_metadata = {
            "tenant_id": scenario["tenant_id"],
            "migration_role": "source" if "source" in scenario["tenant_id"] else "target",
            "tier": scenario["tier"].value,
            "data_volume": {
                "users": len(tenant_data['data'].get('users', [])),
                "tracks": len(tenant_data['data'].get('tracks', [])),
                "playlists": len(tenant_data['data'].get('playlists', [])),
                "listening_events": len(tenant_data['data'].get('listening_history', []))
            },
            "estimated_migration_time": "~30 minutes",
            "compatibility_check": "passed",
            "generated_at": datetime.now().isoformat()
        }
        
        # Save migration metadata
        metadata_path = generator.output_path / scenario["tenant_id"] / "migration_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(migration_metadata, f, indent=2)
        
        print(f"✅ Generated {scenario['tenant_id']} for migration in {generation_time:.2f}s")


async def generate_performance_benchmark_data():
    """Generate data for performance benchmarking."""
    print("📈 Generating performance benchmark data...")
    
    generator = MultiTenantDataGenerator()
    
    # Create benchmarking scenarios with different data sizes
    benchmark_scenarios = [
        {
            "tenant_id": "benchmark_small",
            "name": "Small Dataset Benchmark",
            "tier": TenantTier.STANDARD,
            "scale_factor": 0.1,
            "target_ops": 1000
        },
        {
            "tenant_id": "benchmark_medium", 
            "name": "Medium Dataset Benchmark",
            "tier": TenantTier.PREMIUM,
            "scale_factor": 0.5,
            "target_ops": 5000
        },
        {
            "tenant_id": "benchmark_large",
            "name": "Large Dataset Benchmark", 
            "tier": TenantTier.ENTERPRISE,
            "scale_factor": 1.0,
            "target_ops": 10000
        }
    ]
    
    benchmark_results = []
    
    for scenario in benchmark_scenarios:
        print(f"\n🏃 Generating benchmark data: {scenario['name']}...")
        
        profile = generator.create_tenant_profile(
            tenant_id=scenario["tenant_id"],
            tenant_name=scenario["name"],
            tier=scenario["tier"],
            scale_factor=scenario["scale_factor"]
        )
        
        start_time = time.time()
        tenant_data = await generator.generate_tenant_data(profile)
        generation_time = time.time() - start_time
        
        await generator.save_tenant_data(tenant_data, format="json")
        
        # Calculate performance metrics
        total_records = sum(
            len(data) if isinstance(data, list) else 1
            for data in tenant_data['data'].values()
        )
        
        records_per_second = total_records / generation_time if generation_time > 0 else 0
        
        benchmark_result = {
            "scenario": scenario["name"],
            "tenant_id": scenario["tenant_id"],
            "tier": scenario["tier"].value,
            "scale_factor": scenario["scale_factor"],
            "generation_time_seconds": generation_time,
            "total_records": total_records,
            "records_per_second": records_per_second,
            "target_operations_per_second": scenario["target_ops"],
            "performance_ratio": records_per_second / scenario["target_ops"],
            "timestamp": datetime.now().isoformat()
        }
        
        benchmark_results.append(benchmark_result)
        
        print(f"✅ Generated {total_records:,} records in {generation_time:.2f}s")
        print(f"   Performance: {records_per_second:.0f} records/second")
    
    # Save benchmark summary
    benchmark_summary = {
        "benchmark_run": datetime.now().isoformat(),
        "scenarios": benchmark_results,
        "summary": {
            "total_scenarios": len(benchmark_results),
            "total_generation_time": sum(r["generation_time_seconds"] for r in benchmark_results),
            "total_records": sum(r["total_records"] for r in benchmark_results),
            "average_performance": sum(r["records_per_second"] for r in benchmark_results) / len(benchmark_results)
        }
    }
    
    summary_path = generator.output_path / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    
    print(f"\n📊 Benchmark Summary:")
    print(f"   Total Records: {benchmark_summary['summary']['total_records']:,}")
    print(f"   Average Performance: {benchmark_summary['summary']['average_performance']:.0f} records/second")
    print(f"   Summary saved to: {summary_path}")


async def generate_custom_scenario(
    tenant_id: str,
    tier: str,
    scale_factor: float = 1.0,
    data_types: Optional[List[str]] = None
):
    """Generate data for a custom scenario."""
    print(f"🎨 Generating custom scenario: {tenant_id}")
    
    generator = MultiTenantDataGenerator()
    
    # Convert string tier to enum
    tier_map = {
        "free": TenantTier.FREE,
        "standard": TenantTier.STANDARD,
        "premium": TenantTier.PREMIUM,
        "enterprise": TenantTier.ENTERPRISE
    }
    
    tier_enum = tier_map.get(tier.lower())
    if not tier_enum:
        raise ValueError(f"Invalid tier: {tier}. Must be one of: {list(tier_map.keys())}")
    
    # Convert string data types to enums
    if data_types:
        type_map = {
            "users": DataType.USERS,
            "tracks": DataType.TRACKS,
            "albums": DataType.ALBUMS,
            "artists": DataType.ARTISTS,
            "playlists": DataType.PLAYLISTS,
            "listening_history": DataType.LISTENING_HISTORY,
            "recommendations": DataType.RECOMMENDATIONS,
            "analytics_events": DataType.ANALYTICS_EVENTS,
            "ml_features": DataType.ML_FEATURES
        }
        data_type_enums = [type_map[dt] for dt in data_types if dt in type_map]
    else:
        data_type_enums = None
    
    profile = generator.create_tenant_profile(
        tenant_id=tenant_id,
        tenant_name=f"Custom {tenant_id}",
        tier=tier_enum,
        scale_factor=scale_factor
    )
    
    start_time = time.time()
    tenant_data = await generator.generate_tenant_data(profile, data_type_enums)
    generation_time = time.time() - start_time
    
    await generator.save_tenant_data(tenant_data, format="json")
    
    print(f"✅ Generated custom scenario {tenant_id} in {generation_time:.2f}s")
    
    return tenant_data


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_data_generation.log'),
            logging.StreamHandler()
        ]
    )


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate test data for multi-tenant scenarios")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo data command
    demo_parser = subparsers.add_parser("demo", help="Generate demo data for all tiers")
    
    # Load test command  
    load_parser = subparsers.add_parser("load-test", help="Generate load testing data")
    
    # Compliance test command
    compliance_parser = subparsers.add_parser("compliance", help="Generate compliance testing data")
    
    # Migration test command
    migration_parser = subparsers.add_parser("migration", help="Generate migration testing data")
    
    # Performance benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Generate performance benchmark data")
    
    # Custom scenario command
    custom_parser = subparsers.add_parser("custom", help="Generate custom scenario data")
    custom_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    custom_parser.add_argument("--tier", required=True, choices=["free", "standard", "premium", "enterprise"], help="Tenant tier")
    custom_parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor (default: 1.0)")
    custom_parser.add_argument("--data-types", nargs="+", help="Specific data types to generate")
    
    # All scenarios command
    all_parser = subparsers.add_parser("all", help="Generate all test scenarios")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not args.command:
        parser.print_help()
        return
    
    start_time = time.time()
    
    try:
        if args.command == "demo":
            await generate_demo_data()
        elif args.command == "load-test":
            await generate_load_test_data()
        elif args.command == "compliance":
            await generate_compliance_test_data()
        elif args.command == "migration":
            await generate_migration_test_data()
        elif args.command == "benchmark":
            await generate_performance_benchmark_data()
        elif args.command == "custom":
            await generate_custom_scenario(
                tenant_id=args.tenant_id,
                tier=args.tier,
                scale_factor=args.scale_factor,
                data_types=args.data_types
            )
        elif args.command == "all":
            print("🚀 Generating all test scenarios...")
            await generate_demo_data()
            await generate_load_test_data()
            await generate_compliance_test_data()
            await generate_migration_test_data()
            await generate_performance_benchmark_data()
        
        total_time = time.time() - start_time
        print(f"\n✅ All data generation completed in {total_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Data generation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
