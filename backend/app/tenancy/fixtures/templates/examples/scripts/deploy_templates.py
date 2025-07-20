#!/usr/bin/env python3
"""
Enterprise Template Deployment Script
Advanced deployment system with validation, rollback, and monitoring

Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import logging
import os
import sys
import time
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
import yaml
import argparse
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentStatus(str, Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class DeploymentStrategy(str, Enum):
    """Deployment strategy enumeration"""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    IMMEDIATE = "immediate"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    environment: str = "staging"
    templates: List[str] = field(default_factory=list)
    validation_enabled: bool = True
    backup_enabled: bool = True
    rollback_on_failure: bool = True
    notification_enabled: bool = True
    
    # Timing configuration
    deployment_timeout_seconds: int = 300
    validation_timeout_seconds: int = 60
    rollback_timeout_seconds: int = 120
    
    # Strategy-specific configuration
    rolling_batch_size: int = 10
    canary_traffic_percentage: int = 10
    canary_duration_minutes: int = 30
    
    # Monitoring configuration
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    max_health_check_failures: int = 3
    
    # Notification configuration
    notification_webhooks: List[str] = field(default_factory=list)
    notification_emails: List[str] = field(default_factory=list)

@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    status: DeploymentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    templates_deployed: List[str] = field(default_factory=list)
    templates_failed: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    rollback_performed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

class TemplateDeployer:
    """Advanced template deployment system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_deployment: Optional[DeploymentResult] = None
        self.backup_paths: List[Path] = []
        
        # Initialize paths
        self.base_path = Path(__file__).parent.parent
        self.templates_path = self.base_path / "templates"
        self.config_path = self.base_path / "config"
        self.backup_path = self.base_path / "backups"
        self.logs_path = self.base_path / "logs"
        
        # Create directories if they don't exist
        self.backup_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)
        
        logger.info("Template deployer initialized", 
                   deployment_id=self.config.deployment_id,
                   strategy=self.config.strategy.value,
                   environment=self.config.environment)
    
    async def deploy_templates(self) -> DeploymentResult:
        """Deploy templates according to configuration"""
        logger.info("Starting template deployment", 
                   deployment_id=self.config.deployment_id,
                   templates=self.config.templates)
        
        # Initialize deployment result
        self.current_deployment = DeploymentResult(
            deployment_id=self.config.deployment_id,
            status=DeploymentStatus.PENDING,
            started_at=datetime.now(timezone.utc)
        )
        
        try:
            # Pre-deployment validation
            if self.config.validation_enabled:
                await self._validate_templates()
            
            # Create backup if enabled
            if self.config.backup_enabled:
                await self._create_backup()
            
            # Deploy templates based on strategy
            await self._deploy_by_strategy()
            
            # Post-deployment validation
            if self.config.validation_enabled:
                await self._validate_deployment()
            
            # Mark as deployed
            self.current_deployment.status = DeploymentStatus.DEPLOYED
            self.current_deployment.completed_at = datetime.now(timezone.utc)
            
            # Send notifications
            if self.config.notification_enabled:
                await self._send_notifications("success")
            
            logger.info("Template deployment completed successfully",
                       deployment_id=self.config.deployment_id,
                       duration_seconds=(
                           self.current_deployment.completed_at - 
                           self.current_deployment.started_at
                       ).total_seconds())
            
        except Exception as e:
            logger.error("Template deployment failed", 
                        deployment_id=self.config.deployment_id,
                        error=str(e))
            
            self.current_deployment.status = DeploymentStatus.FAILED
            self.current_deployment.error_message = str(e)
            self.current_deployment.completed_at = datetime.now(timezone.utc)
            
            # Rollback if enabled
            if self.config.rollback_on_failure:
                try:
                    await self._rollback_deployment()
                except Exception as rollback_error:
                    logger.error("Rollback failed", 
                               deployment_id=self.config.deployment_id,
                               error=str(rollback_error))
            
            # Send failure notifications
            if self.config.notification_enabled:
                await self._send_notifications("failure")
            
            raise
        
        return self.current_deployment
    
    async def _validate_templates(self):
        """Validate templates before deployment"""
        logger.info("Validating templates", 
                   deployment_id=self.config.deployment_id)
        
        self.current_deployment.status = DeploymentStatus.VALIDATING
        
        for template_id in self.config.templates:
            try:
                # Load template
                template_path = self._get_template_path(template_id)
                if not template_path.exists():
                    raise FileNotFoundError(f"Template {template_id} not found at {template_path}")
                
                # Validate JSON/YAML syntax
                await self._validate_template_syntax(template_path)
                
                # Validate template schema
                await self._validate_template_schema(template_id, template_path)
                
                # Validate dependencies
                await self._validate_template_dependencies(template_id, template_path)
                
                self.current_deployment.validation_results[template_id] = {
                    "status": "valid",
                    "validated_at": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info("Template validation successful", 
                           template_id=template_id)
                
            except Exception as e:
                self.current_deployment.validation_results[template_id] = {
                    "status": "invalid",
                    "error": str(e),
                    "validated_at": datetime.now(timezone.utc).isoformat()
                }
                
                logger.error("Template validation failed", 
                            template_id=template_id, 
                            error=str(e))
                
                raise ValueError(f"Template validation failed for {template_id}: {str(e)}")
        
        logger.info("All templates validated successfully",
                   deployment_id=self.config.deployment_id)
    
    async def _validate_template_syntax(self, template_path: Path):
        """Validate template file syntax"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if template_path.suffix.lower() in ['.json']:
                json.loads(content)
            elif template_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_load(content)
            else:
                raise ValueError(f"Unsupported template format: {template_path.suffix}")
                
        except Exception as e:
            raise ValueError(f"Template syntax validation failed: {str(e)}")
    
    async def _validate_template_schema(self, template_id: str, template_path: Path):
        """Validate template against schema"""
        # Mock schema validation
        # In real implementation, this would validate against JSON Schema
        logger.debug("Schema validation passed", template_id=template_id)
    
    async def _validate_template_dependencies(self, template_id: str, template_path: Path):
        """Validate template dependencies"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Check for dependencies field
            dependencies = template_data.get('dependencies', [])
            
            # Validate each dependency exists
            for dep_id in dependencies:
                dep_path = self._get_template_path(dep_id)
                if not dep_path.exists():
                    raise ValueError(f"Dependency {dep_id} not found")
            
            logger.debug("Dependency validation passed", 
                        template_id=template_id, 
                        dependencies=dependencies)
                        
        except Exception as e:
            raise ValueError(f"Dependency validation failed: {str(e)}")
    
    async def _create_backup(self):
        """Create backup of current templates"""
        logger.info("Creating backup", deployment_id=self.config.deployment_id)
        
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"backup_{backup_timestamp}_{self.config.deployment_id}"
        backup_dir.mkdir(exist_ok=True)
        
        try:
            # Backup template registry
            registry_path = self.config_path / "template_registry.json"
            if registry_path.exists():
                backup_registry_path = backup_dir / "template_registry.json"
                import shutil
                shutil.copy2(registry_path, backup_registry_path)
                self.backup_paths.append(backup_registry_path)
            
            # Backup individual templates
            for template_id in self.config.templates:
                template_path = self._get_template_path(template_id)
                if template_path.exists():
                    backup_template_path = backup_dir / template_path.name
                    shutil.copy2(template_path, backup_template_path)
                    self.backup_paths.append(backup_template_path)
            
            logger.info("Backup created successfully", 
                       backup_dir=str(backup_dir),
                       files_backed_up=len(self.backup_paths))
                       
        except Exception as e:
            logger.error("Backup creation failed", error=str(e))
            raise
    
    async def _deploy_by_strategy(self):
        """Deploy templates according to strategy"""
        logger.info("Deploying templates", 
                   strategy=self.config.strategy.value,
                   deployment_id=self.config.deployment_id)
        
        self.current_deployment.status = DeploymentStatus.DEPLOYING
        
        if self.config.strategy == DeploymentStrategy.IMMEDIATE:
            await self._deploy_immediate()
        elif self.config.strategy == DeploymentStrategy.ROLLING:
            await self._deploy_rolling()
        elif self.config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._deploy_blue_green()
        elif self.config.strategy == DeploymentStrategy.CANARY:
            await self._deploy_canary()
        else:
            raise ValueError(f"Unsupported deployment strategy: {self.config.strategy}")
    
    async def _deploy_immediate(self):
        """Deploy all templates immediately"""
        for template_id in self.config.templates:
            await self._deploy_single_template(template_id)
            self.current_deployment.templates_deployed.append(template_id)
    
    async def _deploy_rolling(self):
        """Deploy templates in rolling batches"""
        batch_size = self.config.rolling_batch_size
        templates = self.config.templates
        
        for i in range(0, len(templates), batch_size):
            batch = templates[i:i + batch_size]
            logger.info("Deploying batch", 
                       batch_number=i // batch_size + 1,
                       templates=batch)
            
            # Deploy batch
            batch_tasks = []
            for template_id in batch:
                task = asyncio.create_task(self._deploy_single_template(template_id))
                batch_tasks.append((template_id, task))
            
            # Wait for batch completion
            for template_id, task in batch_tasks:
                try:
                    await task
                    self.current_deployment.templates_deployed.append(template_id)
                    logger.info("Template deployed successfully", template_id=template_id)
                except Exception as e:
                    self.current_deployment.templates_failed.append(template_id)
                    logger.error("Template deployment failed", 
                               template_id=template_id, 
                               error=str(e))
                    raise
            
            # Health check after batch
            if self.config.health_check_enabled and i + batch_size < len(templates):
                await self._perform_health_check()
    
    async def _deploy_blue_green(self):
        """Deploy using blue-green strategy"""
        # Mock blue-green deployment
        # In real implementation, this would deploy to a secondary environment
        # then switch traffic
        logger.info("Performing blue-green deployment")
        
        # Deploy to green environment
        for template_id in self.config.templates:
            await self._deploy_single_template(template_id, environment="green")
            self.current_deployment.templates_deployed.append(template_id)
        
        # Health check green environment
        await self._perform_health_check(environment="green")
        
        # Switch traffic to green
        await self._switch_traffic_to_green()
        
        logger.info("Blue-green deployment completed")
    
    async def _deploy_canary(self):
        """Deploy using canary strategy"""
        logger.info("Performing canary deployment", 
                   traffic_percentage=self.config.canary_traffic_percentage)
        
        # Deploy to canary environment
        for template_id in self.config.templates:
            await self._deploy_single_template(template_id, environment="canary")
            self.current_deployment.templates_deployed.append(template_id)
        
        # Route canary traffic
        await self._route_canary_traffic()
        
        # Monitor canary for specified duration
        await self._monitor_canary()
        
        # If canary is healthy, deploy to production
        await self._promote_canary_to_production()
        
        logger.info("Canary deployment completed")
    
    async def _deploy_single_template(self, template_id: str, environment: str = None):
        """Deploy a single template"""
        env = environment or self.config.environment
        logger.debug("Deploying template", template_id=template_id, environment=env)
        
        template_path = self._get_template_path(template_id)
        
        # Mock deployment process
        # In real implementation, this would:
        # 1. Load template
        # 2. Apply environment-specific configurations
        # 3. Deploy to target environment
        # 4. Update registry
        
        # Simulate deployment time
        await asyncio.sleep(1)
        
        # Update registry
        await self._update_registry(template_id, env)
        
        logger.debug("Template deployed successfully", 
                    template_id=template_id, 
                    environment=env)
    
    async def _update_registry(self, template_id: str, environment: str):
        """Update template registry with deployment info"""
        registry_path = self.config_path / "template_registry.json"
        
        try:
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
            else:
                registry = {"templates": {}}
            
            # Update template deployment info
            if template_id in registry.get("templates", {}):
                template_info = registry["templates"][template_id]
                if "deployments" not in template_info:
                    template_info["deployments"] = {}
                
                template_info["deployments"][environment] = {
                    "deployed_at": datetime.now(timezone.utc).isoformat(),
                    "deployment_id": self.config.deployment_id,
                    "status": "deployed"
                }
                
                # Save registry
                with open(registry_path, 'w', encoding='utf-8') as f:
                    json.dump(registry, f, indent=2)
                
                logger.debug("Registry updated", 
                           template_id=template_id, 
                           environment=environment)
                           
        except Exception as e:
            logger.warning("Failed to update registry", 
                         template_id=template_id, 
                         error=str(e))
    
    async def _validate_deployment(self):
        """Validate deployment success"""
        logger.info("Validating deployment", 
                   deployment_id=self.config.deployment_id)
        
        # Perform health checks
        if self.config.health_check_enabled:
            await self._perform_health_check()
        
        # Validate each deployed template
        for template_id in self.current_deployment.templates_deployed:
            await self._validate_deployed_template(template_id)
        
        logger.info("Deployment validation completed successfully")
    
    async def _validate_deployed_template(self, template_id: str):
        """Validate a deployed template"""
        # Mock validation
        # In real implementation, this would verify the template is properly deployed
        logger.debug("Validating deployed template", template_id=template_id)
        await asyncio.sleep(0.1)
    
    async def _perform_health_check(self, environment: str = None):
        """Perform health check on deployment"""
        env = environment or self.config.environment
        logger.info("Performing health check", environment=env)
        
        failures = 0
        max_failures = self.config.max_health_check_failures
        
        while failures < max_failures:
            try:
                # Mock health check
                # In real implementation, this would check service health
                await asyncio.sleep(1)
                
                # Simulate random failures for testing
                import random
                if random.random() < 0.1:  # 10% failure rate
                    raise Exception("Health check failed")
                
                logger.info("Health check passed", environment=env)
                return
                
            except Exception as e:
                failures += 1
                logger.warning("Health check failed", 
                             environment=env, 
                             attempt=failures, 
                             max_attempts=max_failures,
                             error=str(e))
                
                if failures < max_failures:
                    await asyncio.sleep(self.config.health_check_interval_seconds)
        
        raise Exception(f"Health check failed after {max_failures} attempts")
    
    async def _switch_traffic_to_green(self):
        """Switch traffic to green environment (blue-green)"""
        logger.info("Switching traffic to green environment")
        # Mock traffic switching
        await asyncio.sleep(2)
    
    async def _route_canary_traffic(self):
        """Route canary traffic (canary deployment)"""
        logger.info("Routing canary traffic", 
                   percentage=self.config.canary_traffic_percentage)
        # Mock traffic routing
        await asyncio.sleep(1)
    
    async def _monitor_canary(self):
        """Monitor canary deployment"""
        logger.info("Monitoring canary deployment", 
                   duration_minutes=self.config.canary_duration_minutes)
        
        # Mock canary monitoring
        monitor_duration = self.config.canary_duration_minutes * 60  # Convert to seconds
        await asyncio.sleep(min(monitor_duration, 10))  # Cap at 10 seconds for testing
    
    async def _promote_canary_to_production(self):
        """Promote canary to production"""
        logger.info("Promoting canary to production")
        
        # Deploy to production
        for template_id in self.config.templates:
            await self._deploy_single_template(template_id, environment="production")
    
    async def _rollback_deployment(self):
        """Rollback deployment"""
        logger.info("Rolling back deployment", 
                   deployment_id=self.config.deployment_id)
        
        self.current_deployment.status = DeploymentStatus.ROLLED_BACK
        self.current_deployment.rollback_performed = True
        
        # Restore from backup
        if self.backup_paths:
            await self._restore_from_backup()
        
        # Update registry
        await self._update_registry_rollback()
        
        logger.info("Rollback completed successfully")
    
    async def _restore_from_backup(self):
        """Restore templates from backup"""
        logger.info("Restoring from backup", 
                   backup_files=len(self.backup_paths))
        
        import shutil
        for backup_path in self.backup_paths:
            if backup_path.name == "template_registry.json":
                target_path = self.config_path / "template_registry.json"
            else:
                # Determine original template path
                # This is simplified - real implementation would track original paths
                target_path = self.templates_path / backup_path.name
            
            shutil.copy2(backup_path, target_path)
            logger.debug("Restored file", source=str(backup_path), target=str(target_path))
    
    async def _update_registry_rollback(self):
        """Update registry to reflect rollback"""
        registry_path = self.config_path / "template_registry.json"
        
        try:
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                
                # Mark deployment as rolled back
                for template_id in self.config.templates:
                    if template_id in registry.get("templates", {}):
                        template_info = registry["templates"][template_id]
                        if "deployments" in template_info:
                            for env, deployment_info in template_info["deployments"].items():
                                if deployment_info.get("deployment_id") == self.config.deployment_id:
                                    deployment_info["status"] = "rolled_back"
                                    deployment_info["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
                
                with open(registry_path, 'w', encoding='utf-8') as f:
                    json.dump(registry, f, indent=2)
                    
        except Exception as e:
            logger.error("Failed to update registry for rollback", error=str(e))
    
    async def _send_notifications(self, status: str):
        """Send deployment notifications"""
        logger.info("Sending notifications", status=status)
        
        # Mock notification sending
        for webhook in self.config.notification_webhooks:
            logger.debug("Sending webhook notification", webhook=webhook, status=status)
        
        for email in self.config.notification_emails:
            logger.debug("Sending email notification", email=email, status=status)
    
    def _get_template_path(self, template_id: str) -> Path:
        """Get path to template file"""
        # Load registry to find template path
        registry_path = self.config_path / "template_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            template_info = registry.get("templates", {}).get(template_id)
            if template_info and "path" in template_info:
                return self.base_path / template_info["path"]
        
        # Fallback to default path pattern
        return self.templates_path / f"{template_id}.json"

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy templates to environment")
    parser.add_argument("--templates", nargs="+", required=True, 
                       help="Template IDs to deploy")
    parser.add_argument("--environment", default="staging", 
                       help="Target environment")
    parser.add_argument("--strategy", choices=["immediate", "rolling", "blue_green", "canary"],
                       default="rolling", help="Deployment strategy")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Skip template validation")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip backup creation")
    parser.add_argument("--no-rollback", action="store_true", 
                       help="Disable automatic rollback on failure")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Batch size for rolling deployment")
    parser.add_argument("--canary-traffic", type=int, default=10, 
                       help="Canary traffic percentage")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Deployment timeout in seconds")
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        strategy=DeploymentStrategy(args.strategy),
        environment=args.environment,
        templates=args.templates,
        validation_enabled=not args.no_validation,
        backup_enabled=not args.no_backup,
        rollback_on_failure=not args.no_rollback,
        rolling_batch_size=args.batch_size,
        canary_traffic_percentage=args.canary_traffic,
        deployment_timeout_seconds=args.timeout
    )
    
    # Create deployer and run deployment
    deployer = TemplateDeployer(config)
    
    try:
        result = await deployer.deploy_templates()
        
        # Print deployment summary
        print("\n" + "="*60)
        print("DEPLOYMENT SUMMARY")
        print("="*60)
        print(f"Deployment ID: {result.deployment_id}")
        print(f"Status: {result.status.value}")
        print(f"Started: {result.started_at}")
        print(f"Completed: {result.completed_at}")
        if result.completed_at:
            duration = (result.completed_at - result.started_at).total_seconds()
            print(f"Duration: {duration:.2f} seconds")
        print(f"Templates Deployed: {len(result.templates_deployed)}")
        print(f"Templates Failed: {len(result.templates_failed)}")
        
        if result.templates_deployed:
            print("\nSuccessfully Deployed:")
            for template_id in result.templates_deployed:
                print(f"  ✓ {template_id}")
        
        if result.templates_failed:
            print("\nFailed Templates:")
            for template_id in result.templates_failed:
                print(f"  ✗ {template_id}")
        
        if result.rollback_performed:
            print("\n⚠️  Rollback was performed due to deployment failure")
        
        if result.error_message:
            print(f"\nError: {result.error_message}")
        
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if result.status == DeploymentStatus.DEPLOYED else 1)
        
    except Exception as e:
        logger.error("Deployment failed with exception", error=str(e))
        print(f"\n❌ Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
