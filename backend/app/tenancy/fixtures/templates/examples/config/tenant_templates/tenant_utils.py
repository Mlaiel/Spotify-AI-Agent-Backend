#!/usr/bin/env python3
"""
Enterprise Tenant Template Management Utilities
Ultra-Advanced Industrial Multi-Tenant Architecture Scripts

Developed by Expert Team led by Fahed Mlaiel:
- Lead Dev + AI Architect: Fahed Mlaiel - Distributed architecture with integrated ML
- Senior Backend Developer: Python/FastAPI/Django high-performance async architecture  
- ML Engineer: Intelligent recommendations and automatic optimization
- DBA & Data Engineer: Multi-database management with automatic sharding
- Backend Security Specialist: End-to-end encryption and GDPR compliance
- Microservices Architect: Event-Driven patterns with CQRS
"""

import asyncio
import json
import yaml
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Import our enterprise tenant template system
try:
    from . import (
        EnterpriseTenantTemplateManager,
        TenantTier,
        SecurityLevel,
        ComplianceFramework,
        create_enterprise_template_manager,
        calculate_template_cost,
        validate_template_configuration
    )
except ImportError:
    print("Error: Could not import tenant template management system")
    print("Please ensure you're running this script from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tenant_template_management.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TenantTemplateUtility:
    """Enterprise-grade utility for managing tenant templates."""
    
    def __init__(self):
        self.manager: Optional[EnterpriseTenantTemplateManager] = None
        self.templates_directory = Path(__file__).parent
        
    async def initialize(self):
        """Initialize the enterprise template manager."""
        try:
            self.manager = await create_enterprise_template_manager()
            logger.info("✅ Enterprise Tenant Template Manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize template manager: {e}")
            raise
    
    async def create_template_from_yaml(self, yaml_file: str, template_name: Optional[str] = None) -> str:
        """Create a tenant template from YAML configuration."""
        try:
            yaml_path = self.templates_directory / yaml_file
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")
            
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Extract configuration from YAML
            tenant_config = config.get('tenant_template', {})
            tier_name = tenant_config.get('tier', 'STANDARD')
            tier = TenantTier[tier_name]
            
            if not template_name:
                template_name = tenant_config.get('name', f'template_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
            # Create the template
            template = await self.manager.create_tenant_template(
                tier=tier,
                template_name=template_name,
                custom_config=tenant_config
            )
            
            logger.info(f"✅ Created template '{template.name}' (ID: {template.id}) from {yaml_file}")
            return template.id
            
        except Exception as e:
            logger.error(f"❌ Failed to create template from YAML: {e}")
            raise
    
    async def export_template_to_yaml(self, template_id: str, output_file: str):
        """Export a tenant template to YAML format."""
        try:
            yaml_content = await self.manager.export_template_yaml(template_id)
            
            output_path = self.templates_directory / output_file
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(yaml_content)
            
            logger.info(f"✅ Exported template {template_id} to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export template to YAML: {e}")
            raise
    
    async def list_templates(self, tier_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available tenant templates."""
        try:
            templates = await self.manager.list_templates()
            
            if tier_filter:
                tier = TenantTier[tier_filter.upper()]
                templates = [t for t in templates if t.tier == tier]
            
            template_list = []
            for template in templates:
                cost = calculate_template_cost(template)
                template_info = {
                    'id': template.id,
                    'name': template.name,
                    'tier': template.tier.value,
                    'created_at': template.created_at.isoformat(),
                    'monthly_cost_usd': cost,
                    'cpu_cores': template.resource_quotas.cpu_cores,
                    'memory_gb': template.resource_quotas.memory_gb,
                    'storage_gb': template.resource_quotas.storage_gb,
                    'security_level': template.security_config.encryption_level.value,
                    'ai_enabled': template.ai_config.recommendation_engine_enabled
                }
                template_list.append(template_info)
            
            logger.info(f"📋 Found {len(template_list)} templates")
            return template_list
            
        except Exception as e:
            logger.error(f"❌ Failed to list templates: {e}")
            raise
    
    async def validate_template(self, template_id: str) -> Dict[str, Any]:
        """Validate a tenant template configuration."""
        try:
            template = await self.manager.get_template(template_id)
            validation_result = validate_template_configuration(template)
            
            logger.info(f"🔍 Validation result for template {template_id}: {validation_result['status']}")
            
            if validation_result['errors']:
                logger.warning(f"⚠️ Validation errors found: {validation_result['errors']}")
            
            if validation_result['warnings']:
                logger.warning(f"⚠️ Validation warnings: {validation_result['warnings']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Failed to validate template: {e}")
            raise
    
    async def clone_template(self, source_id: str, new_name: str, tier_upgrade: Optional[str] = None) -> str:
        """Clone an existing template with optional tier upgrade."""
        try:
            new_tier = None
            if tier_upgrade:
                new_tier = TenantTier[tier_upgrade.upper()]
            
            new_template = await self.manager.clone_template(source_id, new_name, new_tier)
            
            logger.info(f"✅ Cloned template {source_id} to '{new_name}' (ID: {new_template.id})")
            
            if tier_upgrade:
                logger.info(f"📈 Upgraded tier to {tier_upgrade}")
            
            return new_template.id
            
        except Exception as e:
            logger.error(f"❌ Failed to clone template: {e}")
            raise
    
    async def optimize_template(self, template_id: str) -> Dict[str, Any]:
        """Use AI to optimize template configuration."""
        try:
            optimization_result = await self.manager.optimize_template_with_ai(template_id)
            
            logger.info(f"🤖 AI optimization completed for template {template_id}")
            logger.info(f"💰 Estimated cost savings: ${optimization_result.get('cost_savings', 0)}/month")
            logger.info(f"⚡ Performance improvements: {optimization_result.get('performance_improvements', {})}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize template: {e}")
            raise
    
    async def generate_compliance_report(self, template_id: str) -> Dict[str, Any]:
        """Generate compliance report for a template."""
        try:
            template = await self.manager.get_template(template_id)
            
            compliance_report = {
                'template_id': template_id,
                'template_name': template.name,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'compliance_frameworks': [],
                'security_level': template.security_config.encryption_level.value,
                'audit_requirements': [],
                'data_residency': [],
                'encryption_standards': [],
                'access_controls': [],
                'recommendations': []
            }
            
            # Check each compliance framework
            for framework in template.compliance_config.frameworks_enabled:
                framework_status = {
                    'framework': framework.value,
                    'compliant': True,
                    'requirements_met': [],
                    'requirements_missing': [],
                    'recommendations': []
                }
                
                # Add framework-specific checks
                if framework == ComplianceFramework.GDPR:
                    framework_status['requirements_met'].extend([
                        'Data encryption at rest and in transit',
                        'Right to be forgotten implementation',
                        'Data portability support',
                        'Consent management system'
                    ])
                    if template.security_config.encryption_level in [SecurityLevel.MAXIMUM, SecurityLevel.CLASSIFIED]:
                        framework_status['requirements_met'].append('Strong encryption standards')
                
                elif framework == ComplianceFramework.HIPAA:
                    framework_status['requirements_met'].extend([
                        'PHI encryption',
                        'Access logging',
                        'User authentication',
                        'Backup encryption'
                    ])
                
                compliance_report['compliance_frameworks'].append(framework_status)
            
            logger.info(f"📊 Generated compliance report for template {template_id}")
            return compliance_report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate compliance report: {e}")
            raise
    
    async def batch_create_templates(self, config_file: str):
        """Create multiple templates from batch configuration."""
        try:
            config_path = Path(config_file)
            with open(config_path, 'r', encoding='utf-8') as file:
                batch_config = json.load(file)
            
            created_templates = []
            
            for template_config in batch_config.get('templates', []):
                tier = TenantTier[template_config['tier']]
                template = await self.manager.create_tenant_template(
                    tier=tier,
                    template_name=template_config['name'],
                    custom_config=template_config.get('custom_config', {})
                )
                created_templates.append(template.id)
                logger.info(f"✅ Created template '{template.name}' (ID: {template.id})")
            
            logger.info(f"🎉 Batch created {len(created_templates)} templates")
            return created_templates
            
        except Exception as e:
            logger.error(f"❌ Failed to batch create templates: {e}")
            raise
    
    async def cleanup_old_templates(self, days_old: int = 30, dry_run: bool = True):
        """Clean up old unused templates."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            templates = await self.manager.list_templates()
            
            old_templates = [
                t for t in templates 
                if t.created_at < cutoff_date and not t.in_use
            ]
            
            if dry_run:
                logger.info(f"🔍 DRY RUN: Would delete {len(old_templates)} old templates")
                for template in old_templates:
                    logger.info(f"  - {template.name} (created: {template.created_at})")
            else:
                deleted_count = 0
                for template in old_templates:
                    await self.manager.delete_template(template.id)
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted template '{template.name}'")
                
                logger.info(f"✅ Cleaned up {deleted_count} old templates")
            
            return len(old_templates)
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup templates: {e}")
            raise


async def main():
    """Main CLI interface for tenant template management."""
    parser = argparse.ArgumentParser(
        description="Enterprise Tenant Template Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create template from YAML
  python tenant_utils.py create-from-yaml enterprise.yaml --name "prod_enterprise"
  
  # List all templates
  python tenant_utils.py list --tier ENTERPRISE
  
  # Export template to YAML
  python tenant_utils.py export template_id output.yaml
  
  # Clone and upgrade template
  python tenant_utils.py clone source_id "new_template" --upgrade ENTERPRISE_PLUS
  
  # Validate template
  python tenant_utils.py validate template_id
  
  # Optimize template with AI
  python tenant_utils.py optimize template_id
  
  # Generate compliance report
  python tenant_utils.py compliance template_id
  
  # Batch create templates
  python tenant_utils.py batch-create batch_config.json
  
  # Cleanup old templates
  python tenant_utils.py cleanup --days 30 --execute
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create from YAML command
    create_parser = subparsers.add_parser('create-from-yaml', help='Create template from YAML file')
    create_parser.add_argument('yaml_file', help='YAML configuration file')
    create_parser.add_argument('--name', help='Template name (optional)')
    
    # List templates command
    list_parser = subparsers.add_parser('list', help='List tenant templates')
    list_parser.add_argument('--tier', choices=[t.value for t in TenantTier], help='Filter by tier')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Export template command
    export_parser = subparsers.add_parser('export', help='Export template to YAML')
    export_parser.add_argument('template_id', help='Template ID to export')
    export_parser.add_argument('output_file', help='Output YAML file')
    
    # Clone template command
    clone_parser = subparsers.add_parser('clone', help='Clone existing template')
    clone_parser.add_argument('source_id', help='Source template ID')
    clone_parser.add_argument('new_name', help='New template name')
    clone_parser.add_argument('--upgrade', choices=[t.value for t in TenantTier], help='Upgrade to tier')
    
    # Validate template command
    validate_parser = subparsers.add_parser('validate', help='Validate template configuration')
    validate_parser.add_argument('template_id', help='Template ID to validate')
    
    # Optimize template command
    optimize_parser = subparsers.add_parser('optimize', help='AI-optimize template')
    optimize_parser.add_argument('template_id', help='Template ID to optimize')
    
    # Compliance report command
    compliance_parser = subparsers.add_parser('compliance', help='Generate compliance report')
    compliance_parser.add_argument('template_id', help='Template ID for report')
    compliance_parser.add_argument('--output', help='Output file for report')
    
    # Batch create command
    batch_parser = subparsers.add_parser('batch-create', help='Batch create templates')
    batch_parser.add_argument('config_file', help='JSON configuration file')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old templates')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days old threshold')
    cleanup_parser.add_argument('--execute', action='store_true', help='Actually delete (not dry run)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize utility
    utility = TenantTemplateUtility()
    await utility.initialize()
    
    try:
        if args.command == 'create-from-yaml':
            template_id = await utility.create_template_from_yaml(args.yaml_file, args.name)
            print(f"✅ Created template ID: {template_id}")
        
        elif args.command == 'list':
            templates = await utility.list_templates(args.tier)
            
            if args.format == 'json':
                print(json.dumps(templates, indent=2))
            else:
                # Table format
                print(f"\n{'ID':<20} {'Name':<25} {'Tier':<15} {'CPU':<5} {'RAM':<8} {'Storage':<10} {'Cost/Month':<12}")
                print("-" * 100)
                for t in templates:
                    print(f"{t['id'][:18]:<20} {t['name'][:23]:<25} {t['tier']:<15} {t['cpu_cores']:<5} {t['memory_gb']:<8} {t['storage_gb']:<10} ${t['monthly_cost_usd']:<11}")
        
        elif args.command == 'export':
            await utility.export_template_to_yaml(args.template_id, args.output_file)
            print(f"✅ Exported to {args.output_file}")
        
        elif args.command == 'clone':
            new_id = await utility.clone_template(args.source_id, args.new_name, args.upgrade)
            print(f"✅ Cloned template ID: {new_id}")
        
        elif args.command == 'validate':
            result = await utility.validate_template(args.template_id)
            print(f"Validation Status: {result['status']}")
            if result['errors']:
                print(f"❌ Errors: {result['errors']}")
            if result['warnings']:
                print(f"⚠️ Warnings: {result['warnings']}")
        
        elif args.command == 'optimize':
            result = await utility.optimize_template(args.template_id)
            print(f"✅ Optimization completed")
            print(f"💰 Cost savings: ${result.get('cost_savings', 0)}/month")
            print(f"⚡ Performance improvements: {result.get('performance_improvements', {})}")
        
        elif args.command == 'compliance':
            report = await utility.generate_compliance_report(args.template_id)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"✅ Compliance report saved to {args.output}")
            else:
                print(json.dumps(report, indent=2))
        
        elif args.command == 'batch-create':
            template_ids = await utility.batch_create_templates(args.config_file)
            print(f"✅ Created {len(template_ids)} templates")
        
        elif args.command == 'cleanup':
            count = await utility.cleanup_old_templates(args.days, not args.execute)
            if args.execute:
                print(f"✅ Cleaned up {count} templates")
            else:
                print(f"🔍 Would clean up {count} templates (use --execute to actually delete)")
    
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
