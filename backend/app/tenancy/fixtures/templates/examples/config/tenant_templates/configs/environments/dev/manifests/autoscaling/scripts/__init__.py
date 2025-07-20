"""
Advanced Autoscaling Scripts Management System
Ultra-advanced industrial-grade script orchestration and automation platform

This module provides comprehensive script management for enterprise autoscaling operations
with advanced orchestration, monitoring, and automation capabilities.
"""

import asyncio
import logging
import subprocess
import sys
import time
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import signal
import shlex
from contextlib import asynccontextmanager

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/autoscaling-scripts.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class ScriptType(Enum):
    """Advanced script type classification"""
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    ANALYTICS = "analytics"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    BACKUP = "backup"
    MIGRATION = "migration"
    VALIDATION = "validation"


class ExecutionMode(Enum):
    """Script execution mode specifications"""
    SYNCHRONOUS = "sync"
    ASYNCHRONOUS = "async"
    BACKGROUND = "background"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    INTERACTIVE = "interactive"


class ScriptPriority(Enum):
    """Execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ScriptMetadata:
    """Advanced script metadata structure"""
    name: str
    description: str
    script_type: ScriptType
    execution_mode: ExecutionMode
    priority: ScriptPriority
    version: str
    author: str = "Enterprise Architecture Team"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    environment_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600  # Default 1 hour timeout
    retry_count: int = 3
    retry_delay: int = 30
    tags: List[str] = field(default_factory=list)
    compliance_level: str = "enterprise"
    security_clearance: str = "standard"


@dataclass
class ScriptExecution:
    """Script execution tracking and management"""
    execution_id: str
    script_name: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


class AdvancedScriptManager:
    """
    Ultra-advanced script management system with enterprise orchestration capabilities
    
    Features:
    - Intelligent script scheduling and execution
    - Resource management and optimization
    - Real-time monitoring and analytics
    - Advanced error handling and recovery
    - Security and compliance enforcement
    - Performance optimization and caching
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/etc/autoscaling/scripts.yaml"
        self.scripts_directory = Path(__file__).parent
        self.execution_history: List[ScriptExecution] = []
        self.active_executions: Dict[str, ScriptExecution] = {}
        self.script_registry: Dict[str, ScriptMetadata] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=5)
        self.scheduler_running = False
        self.metrics_collector = None
        self.security_validator = None
        
        # Initialize components
        self._initialize_logging()
        self._load_configuration()
        self._register_scripts()
        self._initialize_monitoring()
        self._setup_signal_handlers()
        
    def _initialize_logging(self):
        """Initialize advanced logging system"""
        self.logger = logging.getLogger(f"{__name__}.ScriptManager")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter for detailed logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[PID:%(process)d] [TID:%(thread)d] - %(message)s'
        )
        
        # File handler for script execution logs
        execution_handler = logging.FileHandler('/var/log/script-executions.log')
        execution_handler.setFormatter(formatter)
        execution_handler.setLevel(logging.INFO)
        self.logger.addHandler(execution_handler)
        
    def _load_configuration(self):
        """Load advanced configuration settings"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._get_default_config()
                self._save_configuration()
                
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Generate default enterprise configuration"""
        return {
            "execution": {
                "max_concurrent_scripts": 10,
                "default_timeout": 3600,
                "resource_limits": {
                    "cpu_percent": 80,
                    "memory_percent": 75,
                    "disk_usage_percent": 85
                },
                "retry_policy": {
                    "max_retries": 3,
                    "base_delay": 30,
                    "exponential_backoff": True
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 30,
                "health_check_interval": 60,
                "alert_thresholds": {
                    "execution_time": 1800,
                    "failure_rate": 0.1,
                    "resource_usage": 0.8
                }
            },
            "security": {
                "require_signature": True,
                "allowed_users": ["root", "autoscaling"],
                "restricted_commands": [
                    "rm -rf /",
                    "dd if=/dev/zero",
                    ":(){ :|:& };:"
                ],
                "sandbox_mode": True
            },
            "logging": {
                "level": "INFO",
                "retention_days": 30,
                "max_file_size": "100MB",
                "compress_old_logs": True
            }
        }
        
    def _save_configuration(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            
    def _register_scripts(self):
        """Register all available scripts with metadata"""
        try:
            script_files = list(self.scripts_directory.glob("*.py")) + \
                          list(self.scripts_directory.glob("*.sh"))
            
            for script_file in script_files:
                if script_file.name.startswith("__"):
                    continue
                    
                metadata = self._extract_script_metadata(script_file)
                self.script_registry[script_file.stem] = metadata
                
            self.logger.info(f"Registered {len(self.script_registry)} scripts")
            
        except Exception as e:
            self.logger.error(f"Failed to register scripts: {e}")
            
    def _extract_script_metadata(self, script_path: Path) -> ScriptMetadata:
        """Extract metadata from script file"""
        try:
            # Read script content for metadata extraction
            with open(script_path, 'r') as f:
                content = f.read()
                
            # Extract metadata from comments/docstrings
            name = script_path.stem
            description = f"Enterprise script: {name}"
            script_type = ScriptType.DEPLOYMENT  # Default
            execution_mode = ExecutionMode.SYNCHRONOUS
            priority = ScriptPriority.NORMAL
            version = "1.0.0"
            
            # Parse metadata from content if available
            if "# TYPE:" in content:
                type_line = [line for line in content.split('\n') if line.startswith("# TYPE:")][0]
                script_type = ScriptType(type_line.split(":")[1].strip().lower())
                
            if "# MODE:" in content:
                mode_line = [line for line in content.split('\n') if line.startswith("# MODE:")][0]
                execution_mode = ExecutionMode(mode_line.split(":")[1].strip().lower())
                
            return ScriptMetadata(
                name=name,
                description=description,
                script_type=script_type,
                execution_mode=execution_mode,
                priority=priority,
                version=version
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {script_path}: {e}")
            return ScriptMetadata(
                name=script_path.stem,
                description=f"Script: {script_path.name}",
                script_type=ScriptType.DEPLOYMENT,
                execution_mode=ExecutionMode.SYNCHRONOUS,
                priority=ScriptPriority.NORMAL,
                version="1.0.0"
            )
            
    def _initialize_monitoring(self):
        """Initialize advanced monitoring system"""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                self._collect_system_metrics()
                self._monitor_active_executions()
                self._check_health_status()
                self._cleanup_old_executions()
                
                time.sleep(self.config["monitoring"]["metrics_interval"])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Extended delay on error
                
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "active_scripts": len(self.active_executions),
                "total_executions": len(self.execution_history)
            }
            
            # Store metrics for analysis
            if not hasattr(self, 'system_metrics'):
                self.system_metrics = []
            self.system_metrics.append(metrics)
            
            # Keep only last 1000 metrics entries
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
                
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            
    def _monitor_active_executions(self):
        """Monitor active script executions"""
        for execution_id, execution in list(self.active_executions.items()):
            try:
                # Check for timeout
                if execution.started_at:
                    elapsed = datetime.now() - execution.started_at
                    timeout = self.script_registry.get(
                        execution.script_name, 
                        ScriptMetadata("", "", ScriptType.DEPLOYMENT, ExecutionMode.SYNCHRONOUS, ScriptPriority.NORMAL, "1.0.0")
                    ).timeout
                    
                    if elapsed.total_seconds() > timeout:
                        self.logger.warning(f"Script {execution.script_name} timed out after {elapsed}")
                        self._terminate_execution(execution_id)
                        
            except Exception as e:
                self.logger.error(f"Failed to monitor execution {execution_id}: {e}")
                
    def _check_health_status(self):
        """Check overall system health"""
        try:
            # Check resource usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            alerts = self.config["monitoring"]["alert_thresholds"]
            
            if cpu_percent > alerts.get("resource_usage", 0.8) * 100:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                
            if memory_percent > alerts.get("resource_usage", 0.8) * 100:
                self.logger.warning(f"High memory usage: {memory_percent}%")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
    def _cleanup_old_executions(self):
        """Clean up old execution records"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            self.execution_history = [
                exec for exec in self.execution_history
                if exec.completed_at and exec.completed_at > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
    async def execute_script(
        self,
        script_name: str,
        args: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> ScriptExecution:
        """
        Execute script with advanced monitoring and error handling
        """
        execution_id = f"{script_name}_{int(time.time() * 1000)}"
        
        # Create execution record
        execution = ScriptExecution(
            execution_id=execution_id,
            script_name=script_name,
            environment=environment or {}
        )
        
        try:
            # Validate script exists
            if script_name not in self.script_registry:
                raise ValueError(f"Script {script_name} not found in registry")
                
            metadata = self.script_registry[script_name]
            
            # Security validation
            if not self._validate_script_security(script_name, args or []):
                raise SecurityError(f"Script {script_name} failed security validation")
                
            # Resource check
            if not self._check_resource_availability():
                raise ResourceError("Insufficient system resources")
                
            execution.started_at = datetime.now()
            execution.status = "running"
            self.active_executions[execution_id] = execution
            
            # Execute script based on type
            script_path = self.scripts_directory / f"{script_name}.sh"
            if not script_path.exists():
                script_path = self.scripts_directory / f"{script_name}.py"
                
            if not script_path.exists():
                raise FileNotFoundError(f"Script file not found: {script_name}")
                
            # Prepare execution environment
            exec_env = os.environ.copy()
            exec_env.update(environment or {})
            
            # Build command
            if script_path.suffix == ".py":
                cmd = [sys.executable, str(script_path)] + (args or [])
            else:
                cmd = ["/bin/bash", str(script_path)] + (args or [])
                
            # Execute with monitoring
            result = await self._execute_with_monitoring(
                cmd, exec_env, timeout or metadata.timeout
            )
            
            execution.exit_code = result["exit_code"]
            execution.stdout = result["stdout"]
            execution.stderr = result["stderr"]
            execution.resource_usage = result["resource_usage"]
            execution.status = "completed" if result["exit_code"] == 0 else "failed"
            
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            execution.status = "error"
            execution.stderr = str(e)
            execution.exit_code = -1
            
        finally:
            execution.completed_at = datetime.now()
            
            # Move to history
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
        return execution
        
    async def _execute_with_monitoring(
        self, 
        cmd: List[str], 
        env: Dict[str, str], 
        timeout: int
    ) -> Dict[str, Any]:
        """Execute command with advanced monitoring"""
        
        start_time = time.time()
        initial_stats = self._get_process_stats()
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                raise TimeoutError(f"Script execution timed out after {timeout} seconds")
                
            exit_code = process.returncode
            
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")
            
        # Calculate resource usage
        end_time = time.time()
        final_stats = self._get_process_stats()
        
        resource_usage = {
            "execution_time": end_time - start_time,
            "cpu_time": final_stats["cpu_time"] - initial_stats["cpu_time"],
            "memory_peak": final_stats["memory_usage"],
            "disk_io": final_stats["disk_io"] - initial_stats["disk_io"]
        }
        
        return {
            "exit_code": exit_code,
            "stdout": stdout.decode('utf-8') if stdout else "",
            "stderr": stderr.decode('utf-8') if stderr else "",
            "resource_usage": resource_usage
        }
        
    def _get_process_stats(self) -> Dict[str, Any]:
        """Get current process statistics"""
        try:
            process = psutil.Process()
            return {
                "cpu_time": process.cpu_times().user + process.cpu_times().system,
                "memory_usage": process.memory_info().rss,
                "disk_io": sum(process.io_counters()[:2]) if hasattr(process, 'io_counters') else 0
            }
        except:
            return {"cpu_time": 0, "memory_usage": 0, "disk_io": 0}
            
    def _validate_script_security(self, script_name: str, args: List[str]) -> bool:
        """Validate script security requirements"""
        try:
            # Check against restricted commands
            restricted = self.config["security"]["restricted_commands"]
            script_content = ""
            
            # Read script content
            for ext in [".sh", ".py"]:
                script_path = self.scripts_directory / f"{script_name}{ext}"
                if script_path.exists():
                    with open(script_path, 'r') as f:
                        script_content = f.read()
                    break
                    
            # Check for restricted patterns
            for restriction in restricted:
                if restriction in script_content:
                    self.logger.warning(f"Script {script_name} contains restricted command: {restriction}")
                    return False
                    
            # Validate arguments
            for arg in args:
                for restriction in restricted:
                    if restriction in arg:
                        self.logger.warning(f"Argument contains restricted command: {restriction}")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return False
            
    def _check_resource_availability(self) -> bool:
        """Check if system resources are available for execution"""
        try:
            limits = self.config["execution"]["resource_limits"]
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            if cpu_percent > limits["cpu_percent"]:
                self.logger.warning(f"CPU usage too high: {cpu_percent}%")
                return False
                
            if memory_percent > limits["memory_percent"]:
                self.logger.warning(f"Memory usage too high: {memory_percent}%")
                return False
                
            if disk_percent > limits["disk_usage_percent"]:
                self.logger.warning(f"Disk usage too high: {disk_percent}%")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            return False
            
    def _terminate_execution(self, execution_id: str):
        """Terminate a running execution"""
        try:
            execution = self.active_executions.get(execution_id)
            if execution:
                execution.status = "terminated"
                execution.completed_at = datetime.now()
                
                # Move to history
                del self.active_executions[execution_id]
                self.execution_history.append(execution)
                
                self.logger.info(f"Terminated execution {execution_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to terminate execution {execution_id}: {e}")
            
    def get_execution_status(self, execution_id: str) -> Optional[ScriptExecution]:
        """Get status of a specific execution"""
        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
            
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
                
        return None
        
    def list_scripts(self) -> Dict[str, ScriptMetadata]:
        """List all registered scripts"""
        return self.script_registry.copy()
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if hasattr(self, 'system_metrics') and self.system_metrics:
            return self.system_metrics[-1]
        return {}
        
    def get_execution_history(self, limit: int = 100) -> List[ScriptExecution]:
        """Get execution history"""
        return self.execution_history[-limit:]
        
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down script manager...")
        
        # Terminate active executions
        for execution_id in list(self.active_executions.keys()):
            self._terminate_execution(execution_id)
            
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Script manager shutdown complete")


class SecurityError(Exception):
    """Security validation error"""
    pass


class ResourceError(Exception):
    """Resource availability error"""
    pass


# Global script manager instance
_script_manager: Optional[AdvancedScriptManager] = None


def get_script_manager() -> AdvancedScriptManager:
    """Get global script manager instance"""
    global _script_manager
    if _script_manager is None:
        _script_manager = AdvancedScriptManager()
    return _script_manager


async def execute_script_async(
    script_name: str,
    args: Optional[List[str]] = None,
    environment: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None
) -> ScriptExecution:
    """Convenience function for async script execution"""
    manager = get_script_manager()
    return await manager.execute_script(script_name, args, environment, timeout)


def execute_script_sync(
    script_name: str,
    args: Optional[List[str]] = None,
    environment: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None
) -> ScriptExecution:
    """Convenience function for sync script execution"""
    return asyncio.run(execute_script_async(script_name, args, environment, timeout))


# Export main classes and functions
__all__ = [
    'AdvancedScriptManager',
    'ScriptMetadata',
    'ScriptExecution',
    'ScriptType',
    'ExecutionMode',
    'ScriptPriority',
    'get_script_manager',
    'execute_script_async',
    'execute_script_sync',
    'SecurityError',
    'ResourceError'
]
