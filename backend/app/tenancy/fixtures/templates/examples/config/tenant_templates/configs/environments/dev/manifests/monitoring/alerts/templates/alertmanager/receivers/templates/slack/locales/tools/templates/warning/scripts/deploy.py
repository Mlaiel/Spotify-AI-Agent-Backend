#!/usr/bin/env python3
"""
Enterprise Deployment Manager for Warning Module
Advanced CI/CD deployment with zero-downtime rollouts
Supports multi-tenant deployments with automated testing and validation
"""

import asyncio
import logging
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import aiofiles
import docker
import kubernetes
from dataclasses import dataclass
import psutil
import redis
from sqlalchemy import create_engine
import boto3

@dataclass
class DeploymentConfig:
    """Advanced deployment configuration"""
    environment: str
    tenant_id: Optional[str]
    version: str
    rollback_version: Optional[str]
    deployment_strategy: str  # blue-green, rolling, canary
    health_check_timeout: int = 300
    max_concurrent_deployments: int = 3
    enable_monitoring: bool = True
    enable_backup: bool = True
    notification_channels: List[str] = None
    
class WarningDeploymentManager:
    """
    Enterprise-grade deployment manager with advanced features:
    - Zero-downtime deployments
    - Multi-tenant isolation
    - Automated rollback on failure
    - Performance monitoring
    - Infrastructure as Code
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.docker_client = docker.from_env()
        self.k8s_client = kubernetes.client.ApiClient()
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])
        self.deployment_history = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration with validation"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['environments', 'deployment', 'monitoring', 'security']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging with multiple handlers"""
        logger = logging.getLogger('deployment_manager')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            'deployment.log', maxBytes=10485760, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Execute deployment with comprehensive monitoring and validation
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Starting deployment {deployment_id}")
            
            # Pre-deployment validation
            await self._validate_pre_deployment(config)
            
            # Create backup if enabled
            if config.enable_backup:
                await self._create_backup(config)
            
            # Execute deployment strategy
            if config.deployment_strategy == "blue-green":
                result = await self._blue_green_deployment(config, deployment_id)
            elif config.deployment_strategy == "rolling":
                result = await self._rolling_deployment(config, deployment_id)
            elif config.deployment_strategy == "canary":
                result = await self._canary_deployment(config, deployment_id)
            else:
                raise ValueError(f"Unknown deployment strategy: {config.deployment_strategy}")
            
            # Post-deployment validation
            await self._validate_post_deployment(config, deployment_id)
            
            # Update deployment history
            self._update_deployment_history(deployment_id, config, result)
            
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            
            # Attempt automatic rollback
            if config.rollback_version:
                await self._rollback(config, deployment_id)
                
            raise
    
    async def _validate_pre_deployment(self, config: DeploymentConfig):
        """Comprehensive pre-deployment validation"""
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        if cpu_usage > 80:
            raise RuntimeError(f"High CPU usage: {cpu_usage}%")
        if memory_usage > 85:
            raise RuntimeError(f"High memory usage: {memory_usage}%")
        if disk_usage > 90:
            raise RuntimeError(f"High disk usage: {disk_usage}%")
        
        # Check database connectivity
        try:
            engine = create_engine(self.config['database_url'])
            with engine.connect() as conn:
                conn.execute("SELECT 1")
        except Exception as e:
            raise RuntimeError(f"Database connectivity check failed: {str(e)}")
        
        # Check Redis connectivity
        try:
            self.redis_client.ping()
        except Exception as e:
            raise RuntimeError(f"Redis connectivity check failed: {str(e)}")
        
        # Validate container images
        await self._validate_container_images(config)
        
        self.logger.info("Pre-deployment validation passed")
    
    async def _validate_container_images(self, config: DeploymentConfig):
        """Validate Docker images are available and secure"""
        
        image_name = f"warning-module:{config.version}"
        
        try:
            # Pull latest image
            self.docker_client.images.pull(image_name)
            
            # Security scan (using Trivy or similar)
            scan_result = subprocess.run([
                'trivy', 'image', '--format', 'json', image_name
            ], capture_output=True, text=True)
            
            if scan_result.returncode == 0:
                scan_data = json.loads(scan_result.stdout)
                high_vulns = sum(1 for vuln in scan_data.get('Results', []) 
                               if vuln.get('Severity') == 'HIGH')
                
                if high_vulns > 0:
                    self.logger.warning(f"Found {high_vulns} high severity vulnerabilities")
                    
        except Exception as e:
            raise RuntimeError(f"Image validation failed: {str(e)}")
    
    async def _blue_green_deployment(self, config: DeploymentConfig, deployment_id: str) -> Dict[str, Any]:
        """
        Blue-Green deployment with zero downtime
        """
        
        # Create new environment (Green)
        green_env = await self._create_green_environment(config, deployment_id)
        
        try:
            # Deploy to Green environment
            await self._deploy_to_environment(green_env, config)
            
            # Run health checks on Green
            health_status = await self._run_health_checks(green_env, config)
            
            if not health_status['healthy']:
                raise RuntimeError(f"Health checks failed: {health_status}")
            
            # Switch traffic to Green
            await self._switch_traffic(green_env, config)
            
            # Cleanup old Blue environment after validation
            await asyncio.sleep(60)  # Grace period
            await self._cleanup_blue_environment(config)
            
            return {
                'deployment_id': deployment_id,
                'strategy': 'blue-green',
                'environment': green_env,
                'health_status': health_status,
                'status': 'success'
            }
            
        except Exception as e:
            # Cleanup failed Green environment
            await self._cleanup_environment(green_env)
            raise
    
    async def _rolling_deployment(self, config: DeploymentConfig, deployment_id: str) -> Dict[str, Any]:
        """
        Rolling deployment with gradual instance replacement
        """
        
        instances = await self._get_running_instances(config)
        total_instances = len(instances)
        
        if total_instances == 0:
            raise RuntimeError("No running instances found")
        
        # Update instances one by one
        updated_instances = []
        
        for i, instance in enumerate(instances):
            self.logger.info(f"Updating instance {i+1}/{total_instances}: {instance['id']}")
            
            try:
                # Create new instance with new version
                new_instance = await self._create_instance(config, deployment_id)
                
                # Wait for new instance to be healthy
                await self._wait_for_instance_health(new_instance, config)
                
                # Remove old instance from load balancer
                await self._remove_from_load_balancer(instance)
                
                # Add new instance to load balancer
                await self._add_to_load_balancer(new_instance)
                
                # Wait for traffic to stabilize
                await asyncio.sleep(30)
                
                # Terminate old instance
                await self._terminate_instance(instance)
                
                updated_instances.append(new_instance)
                
            except Exception as e:
                self.logger.error(f"Failed to update instance {instance['id']}: {str(e)}")
                
                # Rollback updated instances
                for updated_instance in updated_instances:
                    await self._terminate_instance(updated_instance)
                
                raise
        
        return {
            'deployment_id': deployment_id,
            'strategy': 'rolling',
            'updated_instances': len(updated_instances),
            'status': 'success'
        }
    
    async def _canary_deployment(self, config: DeploymentConfig, deployment_id: str) -> Dict[str, Any]:
        """
        Canary deployment with gradual traffic shifting
        """
        
        # Deploy canary instances (10% of total)
        canary_instances = await self._deploy_canary_instances(config, deployment_id)
        
        try:
            # Monitor canary for issues
            canary_metrics = await self._monitor_canary(canary_instances, config)
            
            if not canary_metrics['healthy']:
                raise RuntimeError(f"Canary monitoring failed: {canary_metrics}")
            
            # Gradually increase traffic to canary
            traffic_percentages = [10, 25, 50, 75, 100]
            
            for percentage in traffic_percentages:
                self.logger.info(f"Shifting {percentage}% traffic to canary")
                
                await self._adjust_traffic_split(percentage, config)
                await asyncio.sleep(300)  # Monitor for 5 minutes
                
                metrics = await self._collect_metrics(config)
                if not self._validate_metrics(metrics):
                    raise RuntimeError(f"Metrics validation failed at {percentage}%")
            
            # Complete canary deployment
            await self._complete_canary_deployment(canary_instances, config)
            
            return {
                'deployment_id': deployment_id,
                'strategy': 'canary',
                'canary_instances': len(canary_instances),
                'final_traffic_split': 100,
                'status': 'success'
            }
            
        except Exception as e:
            # Rollback canary
            await self._rollback_canary(canary_instances, config)
            raise
    
    async def _create_backup(self, config: DeploymentConfig):
        """Create comprehensive backup before deployment"""
        
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Database backup
        await self._backup_database(backup_id, config)
        
        # Configuration backup
        await self._backup_configurations(backup_id, config)
        
        # Application state backup
        await self._backup_application_state(backup_id, config)
        
        self.logger.info(f"Backup {backup_id} created successfully")
        return backup_id
    
    async def _rollback(self, config: DeploymentConfig, deployment_id: str):
        """Execute automatic rollback on deployment failure"""
        
        self.logger.info(f"Initiating rollback for deployment {deployment_id}")
        
        rollback_config = DeploymentConfig(
            environment=config.environment,
            tenant_id=config.tenant_id,
            version=config.rollback_version,
            rollback_version=None,
            deployment_strategy="rolling",
            health_check_timeout=config.health_check_timeout
        )
        
        await self.deploy(rollback_config)
        self.logger.info(f"Rollback completed for deployment {deployment_id}")
    
    def _update_deployment_history(self, deployment_id: str, config: DeploymentConfig, result: Dict[str, Any]):
        """Update deployment history for audit and tracking"""
        
        history_entry = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'environment': config.environment,
            'tenant_id': config.tenant_id,
            'version': config.version,
            'strategy': config.deployment_strategy,
            'status': result['status'],
            'duration': result.get('duration'),
            'health_checks': result.get('health_status')
        }
        
        self.deployment_history.append(history_entry)
        
        # Store in Redis for persistence
        self.redis_client.lpush(
            f"deployment_history:{config.environment}",
            json.dumps(history_entry)
        )
        self.redis_client.expire(f"deployment_history:{config.environment}", 86400 * 30)  # 30 days
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get real-time deployment status"""
        
        # Check current running deployments
        running_deployments = await self._get_running_deployments()
        
        if deployment_id in running_deployments:
            return running_deployments[deployment_id]
        
        # Check deployment history
        for entry in self.deployment_history:
            if entry['deployment_id'] == deployment_id:
                return entry
        
        return {'status': 'not_found'}
    
    async def list_deployments(self, environment: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent deployments with filtering"""
        
        deployments = []
        
        if environment:
            # Get from Redis for specific environment
            redis_key = f"deployment_history:{environment}"
            redis_entries = self.redis_client.lrange(redis_key, 0, limit-1)
            
            for entry in redis_entries:
                deployments.append(json.loads(entry))
        else:
            # Get from memory (recent deployments)
            deployments = self.deployment_history[-limit:]
        
        return sorted(deployments, key=lambda x: x['timestamp'], reverse=True)

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = WarningDeploymentManager('deployment_config.yml')
        
        config = DeploymentConfig(
            environment='staging',
            tenant_id='tenant_001',
            version='1.2.0',
            rollback_version='1.1.0',
            deployment_strategy='blue-green'
        )
        
        result = await manager.deploy(config)
        print(f"Deployment result: {result}")
    
    asyncio.run(main())
