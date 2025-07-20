"""
Advanced Monitoring Scripts Module for Spotify AI Agent
========================================================

This module provides industrial-grade automation scripts for monitoring
system deployment, configuration, and maintenance with enterprise-level
reliability and comprehensive automation capabilities.

Core Components:
- Deployment automation with zero-downtime updates
- Alert configuration with ML-powered optimization
- Comprehensive validation with automated reporting
- Performance monitoring and auto-scaling
- Security compliance and audit automation
- Backup and disaster recovery procedures
- Multi-environment orchestration
- Tenant lifecycle management

Features:
- Automated deployment pipelines
- Intelligent configuration management
- Real-time health monitoring
- Predictive maintenance scheduling
- Security scanning and compliance
- Performance optimization automation
- Disaster recovery orchestration
- Complete audit trail logging
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Script execution modes
class ExecutionMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DISASTER_RECOVERY = "disaster_recovery"

class ScriptType(Enum):
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    BACKUP = "backup"
    SECURITY = "security"
    PERFORMANCE = "performance"

@dataclass
class ScriptConfig:
    """Configuration for script execution."""
    name: str
    type: ScriptType
    mode: ExecutionMode
    tenant_id: Optional[str] = None
    environment: str = "dev"
    dry_run: bool = False
    verbose: bool = False
    parallel: bool = False
    timeout: int = 3600
    retry_count: int = 3
    config_path: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class ScriptResult:
    """Result of script execution."""
    success: bool
    exit_code: int
    duration: float
    message: str
    details: Dict[str, Any]
    logs: List[str]
    artifacts: List[str]

class ScriptOrchestrator:
    """Advanced script orchestration and management."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.scripts = {}
        self.results = []
        self._setup_logging()
        self._discover_scripts()
    
    def _setup_logging(self):
        """Configure advanced logging for script execution."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    self.base_path / "script_execution.log",
                    mode='a'
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _discover_scripts(self):
        """Auto-discover available scripts."""
        script_patterns = [
            "deploy_*.sh",
            "setup_*.sh",
            "validate_*.sh",
            "monitor_*.sh",
            "backup_*.sh",
            "security_*.sh",
            "performance_*.sh",
            "maintenance_*.sh"
        ]
        
        for pattern in script_patterns:
            for script_file in self.base_path.glob(pattern):
                script_name = script_file.stem
                self.scripts[script_name] = script_file
                
        self.logger.info(f"Discovered {len(self.scripts)} scripts")
    
    def execute_script(self, script_name: str, config: ScriptConfig) -> ScriptResult:
        """Execute a script with advanced error handling and monitoring."""
        import subprocess
        import time
        
        start_time = time.time()
        
        if script_name not in self.scripts:
            return ScriptResult(
                success=False,
                exit_code=404,
                duration=0,
                message=f"Script not found: {script_name}",
                details={},
                logs=[],
                artifacts=[]
            )
        
        script_path = self.scripts[script_name]
        
        # Build command with arguments
        cmd = [str(script_path)]
        if config.tenant_id:
            cmd.extend(["--tenant", config.tenant_id])
        if config.environment:
            cmd.extend(["--environment", config.environment])
        if config.dry_run:
            cmd.append("--dry-run")
        if config.verbose:
            cmd.append("--verbose")
        
        try:
            # Execute with timeout and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                cwd=self.base_path
            )
            
            duration = time.time() - start_time
            
            return ScriptResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                duration=duration,
                message="Script executed successfully" if result.returncode == 0 else "Script failed",
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": " ".join(cmd)
                },
                logs=result.stdout.split('\n') if result.stdout else [],
                artifacts=[]
            )
            
        except subprocess.TimeoutExpired:
            return ScriptResult(
                success=False,
                exit_code=124,
                duration=config.timeout,
                message="Script execution timed out",
                details={"timeout": config.timeout},
                logs=[],
                artifacts=[]
            )
        except Exception as e:
            return ScriptResult(
                success=False,
                exit_code=1,
                duration=time.time() - start_time,
                message=f"Script execution failed: {str(e)}",
                details={"error": str(e)},
                logs=[],
                artifacts=[]
            )
    
    def execute_pipeline(self, pipeline: List[str], config: ScriptConfig) -> List[ScriptResult]:
        """Execute a pipeline of scripts with dependency management."""
        results = []
        
        for script_name in pipeline:
            self.logger.info(f"Executing pipeline step: {script_name}")
            result = self.execute_script(script_name, config)
            results.append(result)
            
            if not result.success and not config.dry_run:
                self.logger.error(f"Pipeline failed at step: {script_name}")
                break
        
        return results
    
    def get_available_scripts(self) -> Dict[str, str]:
        """Get list of available scripts with descriptions."""
        return {name: str(path) for name, path in self.scripts.items()}

# Global orchestrator instance
orchestrator = ScriptOrchestrator()

# Utility functions for script integration
def execute_deployment(tenant_id: str, environment: str = "dev", dry_run: bool = False) -> ScriptResult:
    """Execute deployment script with standard configuration."""
    config = ScriptConfig(
        name="deploy_monitoring",
        type=ScriptType.DEPLOYMENT,
        mode=ExecutionMode.DEVELOPMENT if environment == "dev" else ExecutionMode.PRODUCTION,
        tenant_id=tenant_id,
        environment=environment,
        dry_run=dry_run,
        verbose=True
    )
    return orchestrator.execute_script("deploy_monitoring", config)

def execute_setup(tenant_id: str, environment: str = "dev") -> ScriptResult:
    """Execute setup script with standard configuration."""
    config = ScriptConfig(
        name="setup_alerts",
        type=ScriptType.CONFIGURATION,
        mode=ExecutionMode.DEVELOPMENT if environment == "dev" else ExecutionMode.PRODUCTION,
        tenant_id=tenant_id,
        environment=environment,
        verbose=True
    )
    return orchestrator.execute_script("setup_alerts", config)

def execute_validation(tenant_id: str = None, generate_report: bool = True) -> ScriptResult:
    """Execute validation script with reporting."""
    config = ScriptConfig(
        name="validate_monitoring",
        type=ScriptType.VALIDATION,
        mode=ExecutionMode.TESTING,
        tenant_id=tenant_id,
        verbose=True
    )
    return orchestrator.execute_script("validate_monitoring", config)

def execute_full_pipeline(tenant_id: str, environment: str = "dev") -> List[ScriptResult]:
    """Execute complete deployment and validation pipeline."""
    config = ScriptConfig(
        name="full_pipeline",
        type=ScriptType.DEPLOYMENT,
        mode=ExecutionMode.DEVELOPMENT if environment == "dev" else ExecutionMode.PRODUCTION,
        tenant_id=tenant_id,
        environment=environment,
        verbose=True
    )
    
    pipeline = [
        "deploy_monitoring",
        "setup_alerts", 
        "validate_monitoring"
    ]
    
    return orchestrator.execute_pipeline(pipeline, config)

# Export public interface
__all__ = [
    'ExecutionMode',
    'ScriptType', 
    'ScriptConfig',
    'ScriptResult',
    'ScriptOrchestrator',
    'orchestrator',
    'execute_deployment',
    'execute_setup',
    'execute_validation',
    'execute_full_pipeline'
]
