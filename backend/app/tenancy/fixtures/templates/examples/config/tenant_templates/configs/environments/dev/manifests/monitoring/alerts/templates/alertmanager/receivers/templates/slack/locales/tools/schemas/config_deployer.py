#!/usr/bin/env python3
"""
Enterprise configuration deployment and management system.

This script provides advanced deployment capabilities with blue/green deployments,
canary releases, rollback functionality, and comprehensive monitoring.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationDeployer:
    """Advanced configuration deployment system."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('DEPLOY_CONFIG_PATH', './deploy_config.yaml')
        self.deployment_history = []
        self.active_deployments = {}
        
        # Load deployment configuration
        self.deploy_config = self._load_deploy_config()
        
    def _load_deploy_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default configuration
                return {
                    "environments": {
                        "development": {
                            "strategy": "direct",
                            "approval_required": False,
                            "rollback_enabled": True
                        },
                        "staging": {
                            "strategy": "blue_green",
                            "approval_required": True,
                            "rollback_enabled": True,
                            "health_check_timeout": 300
                        },
                        "production": {
                            "strategy": "canary",
                            "approval_required": True,
                            "rollback_enabled": True,
                            "canary_percentage": 10,
                            "health_check_timeout": 600
                        }
                    },
                    "notifications": {
                        "slack_webhook": os.getenv('SLACK_WEBHOOK_URL'),
                        "email_recipients": ["ops-team@spotify.com"]
                    }
                }
        except Exception as e:
            logger.error(f"Failed to load deploy config: {e}")
            return {}
    
    async def deploy(self, config_file: str, environment: str, force: bool = False) -> Dict[str, Any]:
        """Deploy configuration to specified environment."""
        deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Validate inputs
            if not Path(config_file).exists():
                raise ValueError(f"Configuration file not found: {config_file}")
            
            env_config = self.deploy_config.get('environments', {}).get(environment)
            if not env_config:
                raise ValueError(f"Environment not configured: {environment}")
            
            # Load configuration
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Start deployment
            deployment = {
                "id": deployment_id,
                "config_file": config_file,
                "environment": environment,
                "strategy": env_config.get('strategy', 'direct'),
                "started_at": datetime.utcnow(),
                "status": "starting",
                "force": force
            }
            
            self.active_deployments[deployment_id] = deployment
            
            # Send start notification
            await self._notify_deployment_start(deployment)
            
            # Execute deployment based on strategy
            if deployment['strategy'] == 'direct':
                result = await self._deploy_direct(deployment, config_data, env_config)
            elif deployment['strategy'] == 'blue_green':
                result = await self._deploy_blue_green(deployment, config_data, env_config)
            elif deployment['strategy'] == 'canary':
                result = await self._deploy_canary(deployment, config_data, env_config)
            else:
                raise ValueError(f"Unknown deployment strategy: {deployment['strategy']}")
            
            # Update deployment status
            deployment.update(result)
            deployment['completed_at'] = datetime.utcnow()
            deployment['duration'] = (deployment['completed_at'] - deployment['started_at']).total_seconds()
            
            # Add to history
            self.deployment_history.append(deployment.copy())
            
            # Send completion notification
            await self._notify_deployment_complete(deployment)
            
            return deployment
            
        except Exception as e:
            # Handle deployment failure
            deployment = self.active_deployments.get(deployment_id, {})
            deployment.update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })
            
            await self._notify_deployment_failed(deployment)
            logger.error(f"Deployment failed: {e}")
            
            return deployment
        
        finally:
            # Cleanup
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
    
    async def _deploy_direct(self, deployment: Dict[str, Any], config_data: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Direct deployment strategy."""
        logger.info(f"Starting direct deployment: {deployment['id']}")
        
        # Validate configuration
        validation_result = await self._validate_configuration(config_data, deployment['environment'])
        if not validation_result['valid']:
            raise ValueError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Apply configuration
        apply_result = await self._apply_configuration(config_data, deployment['environment'])
        
        # Health check
        health_result = await self._health_check(deployment['environment'], env_config.get('health_check_timeout', 300))
        
        return {
            "status": "completed" if health_result['healthy'] else "failed",
            "validation": validation_result,
            "apply_result": apply_result,
            "health_check": health_result
        }
    
    async def _deploy_blue_green(self, deployment: Dict[str, Any], config_data: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Blue/green deployment strategy."""
        logger.info(f"Starting blue/green deployment: {deployment['id']}")
        
        # Validate configuration
        validation_result = await self._validate_configuration(config_data, deployment['environment'])
        if not validation_result['valid']:
            raise ValueError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Deploy to green environment
        green_env = f"{deployment['environment']}_green"
        apply_result = await self._apply_configuration(config_data, green_env)
        
        # Health check green environment
        health_result = await self._health_check(green_env, env_config.get('health_check_timeout', 300))
        
        if not health_result['healthy']:
            raise ValueError("Green environment health check failed")
        
        # Switch traffic to green
        switch_result = await self._switch_traffic(deployment['environment'], green_env)
        
        # Final health check
        final_health = await self._health_check(deployment['environment'], 60)
        
        return {
            "status": "completed" if final_health['healthy'] else "failed",
            "validation": validation_result,
            "apply_result": apply_result,
            "health_check": health_result,
            "switch_result": switch_result,
            "final_health": final_health
        }
    
    async def _deploy_canary(self, deployment: Dict[str, Any], config_data: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Canary deployment strategy."""
        logger.info(f"Starting canary deployment: {deployment['id']}")
        
        canary_percentage = env_config.get('canary_percentage', 10)
        
        # Validate configuration
        validation_result = await self._validate_configuration(config_data, deployment['environment'])
        if not validation_result['valid']:
            raise ValueError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Deploy to canary subset
        canary_result = await self._deploy_canary_subset(config_data, deployment['environment'], canary_percentage)
        
        # Monitor canary for specified duration
        monitor_result = await self._monitor_canary(deployment['environment'], canary_percentage, env_config.get('canary_duration', 300))
        
        if not monitor_result['healthy']:
            # Rollback canary
            rollback_result = await self._rollback_canary(deployment['environment'])
            raise ValueError(f"Canary monitoring failed: {monitor_result['issues']}")
        
        # Promote canary to full deployment
        promote_result = await self._promote_canary(deployment['environment'])
        
        return {
            "status": "completed" if promote_result['success'] else "failed",
            "validation": validation_result,
            "canary_result": canary_result,
            "monitor_result": monitor_result,
            "promote_result": promote_result
        }
    
    async def rollback(self, environment: str, target_version: str = None) -> Dict[str, Any]:
        """Rollback to previous configuration version."""
        logger.info(f"Starting rollback for environment: {environment}")
        
        try:
            # Find target version
            if not target_version:
                # Get previous successful deployment
                previous_deployment = self._get_previous_successful_deployment(environment)
                if not previous_deployment:
                    raise ValueError("No previous successful deployment found")
                target_version = previous_deployment['id']
            
            # Load previous configuration
            previous_config = await self._load_previous_configuration(environment, target_version)
            
            # Execute rollback
            rollback_deployment = {
                "id": f"rollback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "environment": environment,
                "target_version": target_version,
                "started_at": datetime.utcnow(),
                "type": "rollback"
            }
            
            # Apply previous configuration
            apply_result = await self._apply_configuration(previous_config, environment)
            
            # Health check
            health_result = await self._health_check(environment, 300)
            
            rollback_deployment.update({
                "status": "completed" if health_result['healthy'] else "failed",
                "apply_result": apply_result,
                "health_check": health_result,
                "completed_at": datetime.utcnow()
            })
            
            # Notify rollback completion
            await self._notify_rollback_complete(rollback_deployment)
            
            return rollback_deployment
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "environment": environment
            }
    
    async def _validate_configuration(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Validate configuration before deployment."""
        # Implement comprehensive validation logic
        try:
            # Schema validation would go here
            # For now, basic validation
            required_fields = ['name', 'version']
            missing_fields = [field for field in required_fields if field not in config_data]
            
            if missing_fields:
                return {
                    "valid": False,
                    "errors": [f"Missing required field: {field}" for field in missing_fields]
                }
            
            return {"valid": True, "warnings": []}
            
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    async def _apply_configuration(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply configuration to target environment."""
        # Simulate configuration application
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "success": True,
            "applied_configs": len(config_data),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _health_check(self, environment: str, timeout: int) -> Dict[str, Any]:
        """Perform health check on environment."""
        # Simulate health check
        await asyncio.sleep(1)
        
        return {
            "healthy": True,
            "checks": {
                "api_health": "OK",
                "database_health": "OK",
                "redis_health": "OK"
            },
            "response_time_ms": 45
        }
    
    async def _switch_traffic(self, from_env: str, to_env: str) -> Dict[str, Any]:
        """Switch traffic between environments."""
        # Simulate traffic switching
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "from_environment": from_env,
            "to_environment": to_env,
            "switched_at": datetime.utcnow().isoformat()
        }
    
    async def _deploy_canary_subset(self, config_data: Dict[str, Any], environment: str, percentage: int) -> Dict[str, Any]:
        """Deploy configuration to canary subset."""
        await asyncio.sleep(2)
        
        return {
            "success": True,
            "canary_percentage": percentage,
            "deployed_instances": int(100 * percentage / 100),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _monitor_canary(self, environment: str, percentage: int, duration: int) -> Dict[str, Any]:
        """Monitor canary deployment."""
        await asyncio.sleep(duration / 10)  # Simulate monitoring
        
        return {
            "healthy": True,
            "error_rate": 0.001,
            "response_time_p95": 120,
            "monitored_duration": duration
        }
    
    async def _rollback_canary(self, environment: str) -> Dict[str, Any]:
        """Rollback canary deployment."""
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "rolled_back_at": datetime.utcnow().isoformat()
        }
    
    async def _promote_canary(self, environment: str) -> Dict[str, Any]:
        """Promote canary to full deployment."""
        await asyncio.sleep(2)
        
        return {
            "success": True,
            "promoted_at": datetime.utcnow().isoformat()
        }
    
    def _get_previous_successful_deployment(self, environment: str) -> Optional[Dict[str, Any]]:
        """Get previous successful deployment for environment."""
        for deployment in reversed(self.deployment_history):
            if (deployment.get('environment') == environment and 
                deployment.get('status') == 'completed'):
                return deployment
        return None
    
    async def _load_previous_configuration(self, environment: str, version: str) -> Dict[str, Any]:
        """Load previous configuration version."""
        # In real implementation, this would load from version control or backup
        return {
            "name": f"rollback_config_{version}",
            "version": version,
            "environment": environment
        }
    
    async def _notify_deployment_start(self, deployment: Dict[str, Any]):
        """Send deployment start notification."""
        message = f"🚀 Deployment started: {deployment['id']} to {deployment['environment']}"
        logger.info(message)
        # In real implementation, send to Slack/email
    
    async def _notify_deployment_complete(self, deployment: Dict[str, Any]):
        """Send deployment completion notification."""
        status_emoji = "✅" if deployment['status'] == 'completed' else "❌"
        message = f"{status_emoji} Deployment {deployment['status']}: {deployment['id']}"
        logger.info(message)
    
    async def _notify_deployment_failed(self, deployment: Dict[str, Any]):
        """Send deployment failure notification."""
        message = f"❌ Deployment failed: {deployment['id']} - {deployment.get('error', 'Unknown error')}"
        logger.error(message)
    
    async def _notify_rollback_complete(self, rollback: Dict[str, Any]):
        """Send rollback completion notification."""
        status_emoji = "✅" if rollback['status'] == 'completed' else "❌"
        message = f"{status_emoji} Rollback {rollback['status']}: {rollback['environment']}"
        logger.info(message)


async def main():
    """CLI interface for configuration deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration deployment tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy configuration')
    deploy_parser.add_argument('config_file', help='Configuration file to deploy')
    deploy_parser.add_argument('environment', help='Target environment')
    deploy_parser.add_argument('--force', action='store_true', help='Force deployment')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback configuration')
    rollback_parser.add_argument('environment', help='Environment to rollback')
    rollback_parser.add_argument('--version', help='Target version to rollback to')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show deployment status')
    status_parser.add_argument('--environment', help='Filter by environment')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show deployment history')
    history_parser.add_argument('--limit', type=int, default=10, help='Number of deployments to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    deployer = ConfigurationDeployer()
    
    try:
        if args.command == 'deploy':
            result = await deployer.deploy(args.config_file, args.environment, args.force)
            print(f"Deployment result: {result['status']}")
            if result['status'] == 'failed':
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        elif args.command == 'rollback':
            result = await deployer.rollback(args.environment, args.version)
            print(f"Rollback result: {result['status']}")
            if result['status'] == 'failed':
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        elif args.command == 'status':
            active = deployer.active_deployments
            if active:
                print("Active deployments:")
                for deployment_id, deployment in active.items():
                    print(f"  {deployment_id}: {deployment['status']} ({deployment['environment']})")
            else:
                print("No active deployments")
        
        elif args.command == 'history':
            history = deployer.deployment_history[-args.limit:]
            if history:
                print(f"Recent deployments (last {len(history)}):")
                for deployment in history:
                    print(f"  {deployment['id']}: {deployment['status']} ({deployment['environment']})")
            else:
                print("No deployment history")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
