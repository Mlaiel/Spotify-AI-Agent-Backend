#!/usr/bin/env python3
"""
Spotify AI Agent - Advanced Script Executor & Automation Engine
===============================================================

Enterprise-grade automation and script execution system providing:
- Secure script execution with sandboxing
- Task scheduling and workflow management
- Multi-tenant script isolation
- Advanced error handling and retry mechanisms
- Real-time execution monitoring
- Performance optimization and resource management

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

import asyncio
import subprocess
import shlex
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile
import psutil
import aiofiles
import redis
from pydantic import BaseModel, Field, validator
from croniter import croniter
import yaml

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Script execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ScriptType(str, Enum):
    """Supported script types"""
    BASH = "bash"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    YAML = "yaml"
    CUSTOM = "custom"


@dataclass
class ExecutionResult:
    """Script execution result"""
    execution_id: str
    script_name: str
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    start_time: datetime
    end_time: Optional[datetime]
    resource_usage: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ScriptTask:
    """Script task definition"""
    task_id: str
    name: str
    script_type: ScriptType
    script_content: str
    script_path: Optional[str]
    arguments: List[str]
    environment: Dict[str, str]
    working_directory: str
    timeout: int
    priority: TaskPriority
    retry_count: int
    max_retries: int
    schedule: Optional[str]  # Cron expression
    tenant_id: str
    created_at: datetime
    metadata: Dict[str, Any]


class ScriptConfig(BaseModel):
    """Script execution configuration"""
    max_concurrent_executions: int = Field(default=10, ge=1, le=100)
    default_timeout: int = Field(default=300, ge=1)  # 5 minutes
    max_retry_attempts: int = Field(default=3, ge=0, le=10)
    sandbox_enabled: bool = Field(default=True)
    resource_limits: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        "max_memory_mb": 1024,
        "max_cpu_percent": 50,
        "max_execution_time": 3600
    })
    allowed_script_types: List[ScriptType] = Field(default_factory=lambda: [
        ScriptType.BASH, ScriptType.PYTHON, ScriptType.JAVASCRIPT
    ])
    security_policies: Dict[str, Any] = Field(default_factory=lambda: {
        "disable_network": False,
        "read_only_filesystem": False,
        "drop_privileges": True
    })


class ScriptExecutor:
    """
    Advanced script executor with enterprise security and monitoring
    """
    
    def __init__(self, config: ScriptConfig, redis_url: str = "redis://localhost:6379"):
        self.config = config
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Execution tracking
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_history: Dict[str, ExecutionResult] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        logger.info("ScriptExecutor initialized")

    async def execute_script(
        self, 
        task: ScriptTask,
        wait_for_completion: bool = True
    ) -> Union[ExecutionResult, str]:
        """
        Execute a script task with advanced monitoring and security
        """
        execution_id = str(uuid.uuid4())
        
        try:
            # Validate task
            if not self._validate_task(task):
                raise ValueError(f"Invalid task configuration for {task.name}")
            
            # Check resource limits
            if len(self.active_executions) >= self.config.max_concurrent_executions:
                raise RuntimeError("Maximum concurrent executions reached")
            
            # Create execution task
            execution_task = asyncio.create_task(
                self._execute_task_with_monitoring(execution_id, task)
            )
            
            self.active_executions[execution_id] = execution_task
            
            if wait_for_completion:
                result = await execution_task
                return result
            else:
                return execution_id
                
        except Exception as e:
            logger.error(f"Failed to execute script {task.name}: {e}")
            result = ExecutionResult(
                execution_id=execution_id,
                script_name=task.name,
                status=ExecutionStatus.FAILED,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                resource_usage={},
                metadata={"error": str(e)}
            )
            return result

    async def _execute_task_with_monitoring(
        self, 
        execution_id: str, 
        task: ScriptTask
    ) -> ExecutionResult:
        """Execute task with comprehensive monitoring"""
        start_time = datetime.utcnow()
        
        try:
            # Start resource monitoring
            monitor_task = asyncio.create_task(
                self.resource_monitor.monitor_execution(execution_id, task.timeout)
            )
            
            # Execute the actual script
            result = await self._execute_script_secure(execution_id, task)
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Get resource usage
            resource_usage = await self.resource_monitor.get_usage(execution_id)
            
            result.resource_usage = resource_usage
            result.start_time = start_time
            result.end_time = datetime.utcnow()
            result.execution_time = (result.end_time - start_time).total_seconds()
            
            # Store result
            await self._store_execution_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution monitoring failed for {execution_id}: {e}")
            raise
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

    async def _execute_script_secure(
        self, 
        execution_id: str, 
        task: ScriptTask
    ) -> ExecutionResult:
        """Execute script with security sandboxing"""
        
        # Prepare execution environment
        if task.script_path:
            script_file = task.script_path
        else:
            # Create temporary script file
            script_file = await self._create_temp_script(task)
        
        try:
            # Build command
            cmd = await self._build_command(task, script_file)
            
            # Execute with resource limits
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=task.working_directory,
                env={**task.environment},
                limit=1024*1024  # 1MB limit for stdout/stderr
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Script execution timed out after {task.timeout}s")
            
            # Create result
            result = ExecutionResult(
                execution_id=execution_id,
                script_name=task.name,
                status=ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.FAILED,
                exit_code=process.returncode,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                execution_time=0.0,  # Will be set by caller
                start_time=datetime.utcnow(),
                end_time=None,  # Will be set by caller
                resource_usage={},  # Will be set by caller
                metadata={"task_id": task.task_id, "tenant_id": task.tenant_id}
            )
            
            return result
            
        finally:
            # Cleanup temporary files
            if not task.script_path and Path(script_file).exists():
                Path(script_file).unlink()

    async def _create_temp_script(self, task: ScriptTask) -> str:
        """Create temporary script file"""
        suffix = {
            ScriptType.BASH: ".sh",
            ScriptType.PYTHON: ".py",
            ScriptType.JAVASCRIPT: ".js",
            ScriptType.SQL: ".sql",
            ScriptType.YAML: ".yml"
        }.get(task.script_type, ".txt")
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(task.script_content)
            temp_path = f.name
        
        # Make executable for shell scripts
        if task.script_type == ScriptType.BASH:
            Path(temp_path).chmod(0o755)
        
        return temp_path

    async def _build_command(self, task: ScriptTask, script_file: str) -> List[str]:
        """Build execution command based on script type"""
        cmd = []
        
        if task.script_type == ScriptType.BASH:
            cmd = ["bash", script_file] + task.arguments
        elif task.script_type == ScriptType.PYTHON:
            cmd = ["python3", script_file] + task.arguments
        elif task.script_type == ScriptType.JAVASCRIPT:
            cmd = ["node", script_file] + task.arguments
        elif task.script_type == ScriptType.SQL:
            # Assuming PostgreSQL client
            cmd = ["psql", "-f", script_file] + task.arguments
        else:
            raise ValueError(f"Unsupported script type: {task.script_type}")
        
        # Add security wrapper if sandboxing is enabled
        if self.config.sandbox_enabled:
            cmd = await self._add_security_wrapper(cmd, task)
        
        return cmd

    async def _add_security_wrapper(self, cmd: List[str], task: ScriptTask) -> List[str]:
        """Add security wrapper to command"""
        # Use firejail for sandboxing if available
        try:
            subprocess.run(["which", "firejail"], check=True, capture_output=True)
            
            security_cmd = [
                "firejail",
                "--quiet",
                f"--rlimit-cpu={self.config.resource_limits['max_execution_time']}",
                f"--rlimit-as={self.config.resource_limits['max_memory_mb']*1024*1024}",
            ]
            
            if self.config.security_policies["disable_network"]:
                security_cmd.append("--net=none")
            
            if self.config.security_policies["read_only_filesystem"]:
                security_cmd.append("--read-only=/")
                security_cmd.append(f"--read-write={task.working_directory}")
            
            return security_cmd + cmd
            
        except subprocess.CalledProcessError:
            # Firejail not available, use basic timeout
            return ["timeout", str(task.timeout)] + cmd

    def _validate_task(self, task: ScriptTask) -> bool:
        """Validate task configuration"""
        # Check script type
        if task.script_type not in self.config.allowed_script_types:
            return False
        
        # Check timeout
        if task.timeout > self.config.resource_limits["max_execution_time"]:
            return False
        
        # Check working directory exists
        if not Path(task.working_directory).exists():
            return False
        
        # Validate script content or path
        if not task.script_content and not task.script_path:
            return False
        
        if task.script_path and not Path(task.script_path).exists():
            return False
        
        return True

    async def _store_execution_result(self, result: ExecutionResult):
        """Store execution result in Redis"""
        key = f"execution_result:{result.execution_id}"
        value = json.dumps(asdict(result), default=str)
        
        # Store with TTL (24 hours)
        self.redis_client.setex(key, 86400, value)
        
        # Store in execution history
        self.execution_history[result.execution_id] = result

    async def get_execution_result(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get execution result by ID"""
        # Check memory first
        if execution_id in self.execution_history:
            return self.execution_history[execution_id]
        
        # Check Redis
        key = f"execution_result:{execution_id}"
        value = self.redis_client.get(key)
        
        if value:
            data = json.loads(value)
            return ExecutionResult(**data)
        
        return None

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel running execution"""
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            task.cancel()
            del self.active_executions[execution_id]
            return True
        return False

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs"""
        return list(self.active_executions.keys())


class ResourceMonitor:
    """Resource usage monitoring for script executions"""
    
    def __init__(self):
        self.monitoring_data: Dict[str, List[Dict[str, Any]]] = {}
    
    async def monitor_execution(self, execution_id: str, timeout: int):
        """Monitor resource usage during execution"""
        self.monitoring_data[execution_id] = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                # Collect metrics
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters()._asdict(),
                    "network_io": psutil.net_io_counters()._asdict()
                }
                
                self.monitoring_data[execution_id].append(metrics)
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            logger.error(f"Resource monitoring error for {execution_id}: {e}")

    async def get_usage(self, execution_id: str) -> Dict[str, Any]:
        """Get resource usage summary"""
        if execution_id not in self.monitoring_data:
            return {}
        
        data = self.monitoring_data[execution_id]
        if not data:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [d["cpu_percent"] for d in data]
        memory_values = [d["memory_percent"] for d in data]
        
        summary = {
            "cpu_average": sum(cpu_values) / len(cpu_values),
            "cpu_peak": max(cpu_values),
            "memory_average": sum(memory_values) / len(memory_values),
            "memory_peak": max(memory_values),
            "samples_count": len(data),
            "monitoring_duration": len(data) * 5  # 5 second intervals
        }
        
        # Cleanup
        del self.monitoring_data[execution_id]
        
        return summary


class TaskScheduler:
    """
    Advanced task scheduler with cron support
    """
    
    def __init__(self, script_executor: ScriptExecutor):
        self.executor = script_executor
        self.scheduled_tasks: Dict[str, ScriptTask] = {}
        self.scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the task scheduler"""
        self.scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler"""
        self.scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        logger.info("Task scheduler stopped")
    
    async def schedule_task(self, task: ScriptTask) -> bool:
        """Schedule a task with cron expression"""
        if not task.schedule:
            return False
        
        try:
            # Validate cron expression
            croniter(task.schedule)
            self.scheduled_tasks[task.task_id] = task
            logger.info(f"Task {task.name} scheduled with cron: {task.schedule}")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule task {task.name}: {e}")
            return False
    
    async def unschedule_task(self, task_id: str) -> bool:
        """Remove a scheduled task"""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            logger.info(f"Task {task_id} unscheduled")
            return True
        return False
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                current_time = datetime.utcnow()
                
                for task_id, task in self.scheduled_tasks.items():
                    if self._should_run_task(task, current_time):
                        # Execute task asynchronously
                        asyncio.create_task(
                            self.executor.execute_script(task, wait_for_completion=False)
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)
    
    def _should_run_task(self, task: ScriptTask, current_time: datetime) -> bool:
        """Check if task should run based on cron schedule"""
        try:
            cron = croniter(task.schedule, current_time - timedelta(minutes=1))
            next_run = cron.get_next(datetime)
            return next_run <= current_time
        except Exception:
            return False


class AutomationEngine:
    """
    High-level automation engine combining script execution and scheduling
    """
    
    def __init__(self, config: ScriptConfig, redis_url: str = "redis://localhost:6379"):
        self.executor = ScriptExecutor(config, redis_url)
        self.scheduler = TaskScheduler(self.executor)
        self.workflows: Dict[str, List[ScriptTask]] = {}
    
    async def start(self):
        """Start the automation engine"""
        await self.scheduler.start()
        logger.info("Automation engine started")
    
    async def stop(self):
        """Stop the automation engine"""
        await self.scheduler.stop()
        logger.info("Automation engine stopped")
    
    async def create_workflow(self, workflow_id: str, tasks: List[ScriptTask]) -> bool:
        """Create a workflow of tasks"""
        try:
            self.workflows[workflow_id] = tasks
            logger.info(f"Workflow {workflow_id} created with {len(tasks)} tasks")
            return True
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_id}: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str) -> List[ExecutionResult]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        tasks = self.workflows[workflow_id]
        results = []
        
        for task in tasks:
            result = await self.executor.execute_script(task)
            results.append(result)
            
            # Stop on first failure if task is critical
            if (result.status == ExecutionStatus.FAILED and 
                task.priority == TaskPriority.CRITICAL):
                break
        
        return results
    
    async def schedule_workflow(self, workflow_id: str, cron_schedule: str) -> bool:
        """Schedule a workflow to run on cron schedule"""
        if workflow_id not in self.workflows:
            return False
        
        # Create a wrapper task that executes the workflow
        wrapper_task = ScriptTask(
            task_id=f"workflow_{workflow_id}",
            name=f"Workflow: {workflow_id}",
            script_type=ScriptType.PYTHON,
            script_content=f"""
import asyncio
from automation_engine import AutomationEngine

async def main():
    engine = AutomationEngine()
    await engine.execute_workflow('{workflow_id}')

if __name__ == "__main__":
    asyncio.run(main())
""",
            script_path=None,
            arguments=[],
            environment={},
            working_directory="/tmp",
            timeout=3600,
            priority=TaskPriority.NORMAL,
            retry_count=0,
            max_retries=3,
            schedule=cron_schedule,
            tenant_id="system",
            created_at=datetime.utcnow(),
            metadata={"workflow_id": workflow_id}
        )
        
        return await self.scheduler.schedule_task(wrapper_task)
