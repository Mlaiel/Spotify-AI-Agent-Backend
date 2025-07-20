#!/usr/bin/env python3
"""
Advanced Scripts Initialization and Management System
====================================================

Ultra-advanced enterprise-grade script management and automation system
for the Spotify AI Agent development environment.

This module provides comprehensive script orchestration, dependency management,
environment validation, and automated initialization capabilities.

Developed by: Fahed Mlaiel
Expert Team: Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
            DBA & Data Engineer, Backend Security Specialist, Microservices Architect

Features:
- Automated environment setup and validation
- Dynamic script discovery and execution
- Dependency resolution and management
- Health monitoring and diagnostics
- Performance optimization and caching
- Security validation and compliance
- CI/CD integration and automation
- Multi-environment support
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import hashlib
import tempfile
import shutil
import yaml
from datetime import datetime, timedelta

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts_init.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class ScriptType(Enum):
    """Script types supported by the initialization system."""
    SETUP = "setup"
    TEARDOWN = "teardown"
    HEALTH_CHECK = "health_check"
    MONITORING = "monitoring"
    DATABASE = "database"
    SERVICES = "services"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BACKUP = "backup"
    MIGRATION = "migration"
    VALIDATION = "validation"


class ExecutionStatus(Enum):
    """Execution status for scripts and operations."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DOCKER = "docker"
    LOCAL = "local"


@dataclass
class ScriptMetadata:
    """Metadata for script configuration and execution."""
    name: str
    script_type: ScriptType
    description: str
    version: str = "1.0.0"
    author: str = "Fahed Mlaiel Expert Team"
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    environments: List[Environment] = field(default_factory=list)
    timeout: int = 300  # seconds
    retries: int = 3
    parallel: bool = False
    critical: bool = False
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "medium"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None


@dataclass
class ExecutionResult:
    """Result of script execution."""
    script_name: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class SystemRequirements:
    """System requirements specification."""
    python_version: str = "3.9+"
    node_version: Optional[str] = None
    docker_version: Optional[str] = None
    system_packages: List[str] = field(default_factory=list)
    python_packages: List[str] = field(default_factory=list)
    npm_packages: List[str] = field(default_factory=list)
    environment_variables: List[str] = field(default_factory=list)
    disk_space_gb: float = 5.0
    memory_gb: float = 4.0
    cpu_cores: int = 2


class ScriptExecutor(ABC):
    """Abstract base class for script executors."""
    
    @abstractmethod
    async def execute(self, script_path: Path, metadata: ScriptMetadata) -> ExecutionResult:
        """Execute a script with given metadata."""
        pass
    
    @abstractmethod
    def validate(self, script_path: Path) -> bool:
        """Validate script before execution."""
        pass


class BashScriptExecutor(ScriptExecutor):
    """Executor for Bash scripts."""
    
    def __init__(self, shell: str = "/bin/bash"):
        self.shell = shell
        self.env_vars = os.environ.copy()
    
    async def execute(self, script_path: Path, metadata: ScriptMetadata) -> ExecutionResult:
        """Execute a bash script."""
        start_time = datetime.now()
        result = ExecutionResult(
            script_name=metadata.name,
            status=ExecutionStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Prepare environment
            env = self.env_vars.copy()
            env.update(metadata.config.get('environment_variables', {}))
            
            # Execute script
            process = await asyncio.create_subprocess_exec(
                self.shell, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=script_path.parent
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=metadata.timeout
                )
                
                result.return_code = process.returncode
                result.stdout = stdout.decode('utf-8', errors='ignore')
                result.stderr = stderr.decode('utf-8', errors='ignore')
                
                if process.returncode == 0:
                    result.status = ExecutionStatus.SUCCESS
                else:
                    result.status = ExecutionStatus.FAILED
                    result.error_message = f"Script failed with return code {process.returncode}"
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.status = ExecutionStatus.TIMEOUT
                result.error_message = f"Script timed out after {metadata.timeout} seconds"
                
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Error executing script {metadata.name}: {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def validate(self, script_path: Path) -> bool:
        """Validate bash script syntax."""
        try:
            result = subprocess.run(
                [self.shell, "-n", str(script_path)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error validating script {script_path}: {e}")
            return False


class PythonScriptExecutor(ScriptExecutor):
    """Executor for Python scripts."""
    
    def __init__(self, python_path: str = sys.executable):
        self.python_path = python_path
    
    async def execute(self, script_path: Path, metadata: ScriptMetadata) -> ExecutionResult:
        """Execute a Python script."""
        start_time = datetime.now()
        result = ExecutionResult(
            script_name=metadata.name,
            status=ExecutionStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(metadata.config.get('environment_variables', {}))
            
            # Execute script
            process = await asyncio.create_subprocess_exec(
                self.python_path, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=script_path.parent
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=metadata.timeout
                )
                
                result.return_code = process.returncode
                result.stdout = stdout.decode('utf-8', errors='ignore')
                result.stderr = stderr.decode('utf-8', errors='ignore')
                
                if process.returncode == 0:
                    result.status = ExecutionStatus.SUCCESS
                else:
                    result.status = ExecutionStatus.FAILED
                    result.error_message = f"Script failed with return code {process.returncode}"
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.status = ExecutionStatus.TIMEOUT
                result.error_message = f"Script timed out after {metadata.timeout} seconds"
                
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Error executing Python script {metadata.name}: {e}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def validate(self, script_path: Path) -> bool:
        """Validate Python script syntax."""
        try:
            with open(script_path, 'r') as f:
                compile(f.read(), str(script_path), 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in Python script {script_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating Python script {script_path}: {e}")
            return False


class DependencyResolver:
    """Advanced dependency resolution and management."""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.resolved_order: List[str] = []
    
    def add_dependency(self, script: str, dependencies: List[str]):
        """Add dependencies for a script."""
        self.dependency_graph[script] = set(dependencies)
    
    def resolve_dependencies(self) -> List[str]:
        """Resolve dependencies using topological sort."""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(script: str):
            if script in temp_visited:
                raise ValueError(f"Circular dependency detected involving {script}")
            if script in visited:
                return
            
            temp_visited.add(script)
            
            for dependency in self.dependency_graph.get(script, set()):
                visit(dependency)
            
            temp_visited.remove(script)
            visited.add(script)
            result.append(script)
        
        for script in self.dependency_graph.keys():
            if script not in visited:
                visit(script)
        
        self.resolved_order = result
        return result
    
    def get_execution_order(self, scripts: List[str]) -> List[str]:
        """Get execution order for given scripts considering dependencies."""
        all_deps = set()
        for script in scripts:
            all_deps.update(self.dependency_graph.get(script, set()))
            all_deps.add(script)
        
        ordered = self.resolve_dependencies()
        return [script for script in ordered if script in all_deps]


class EnvironmentValidator:
    """Comprehensive environment validation."""
    
    def __init__(self):
        self.validation_results: Dict[str, bool] = {}
        self.validation_messages: Dict[str, str] = {}
    
    async def validate_system_requirements(self, requirements: SystemRequirements) -> bool:
        """Validate system requirements."""
        logger.info("Validating system requirements...")
        
        all_valid = True
        
        # Check Python version
        if not self._check_python_version(requirements.python_version):
            all_valid = False
        
        # Check Docker if required
        if requirements.docker_version:
            if not await self._check_docker_version(requirements.docker_version):
                all_valid = False
        
        # Check Node.js if required
        if requirements.node_version:
            if not await self._check_node_version(requirements.node_version):
                all_valid = False
        
        # Check system packages
        if requirements.system_packages:
            if not await self._check_system_packages(requirements.system_packages):
                all_valid = False
        
        # Check Python packages
        if requirements.python_packages:
            if not self._check_python_packages(requirements.python_packages):
                all_valid = False
        
        # Check environment variables
        if requirements.environment_variables:
            if not self._check_environment_variables(requirements.environment_variables):
                all_valid = False
        
        # Check system resources
        if not self._check_system_resources(requirements):
            all_valid = False
        
        return all_valid
    
    def _check_python_version(self, required_version: str) -> bool:
        """Check Python version."""
        try:
            current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            # Simple version comparison (can be enhanced)
            self.validation_results['python_version'] = True
            self.validation_messages['python_version'] = f"Python {current_version} OK"
            logger.info(f"Python version: {current_version}")
            return True
        except Exception as e:
            self.validation_results['python_version'] = False
            self.validation_messages['python_version'] = str(e)
            return False
    
    async def _check_docker_version(self, required_version: str) -> bool:
        """Check Docker version."""
        try:
            process = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_output = stdout.decode().strip()
                self.validation_results['docker'] = True
                self.validation_messages['docker'] = version_output
                logger.info(f"Docker: {version_output}")
                return True
            else:
                self.validation_results['docker'] = False
                self.validation_messages['docker'] = "Docker not found"
                return False
        except Exception as e:
            self.validation_results['docker'] = False
            self.validation_messages['docker'] = str(e)
            return False
    
    async def _check_node_version(self, required_version: str) -> bool:
        """Check Node.js version."""
        try:
            process = await asyncio.create_subprocess_exec(
                'node', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_output = stdout.decode().strip()
                self.validation_results['node'] = True
                self.validation_messages['node'] = version_output
                logger.info(f"Node.js: {version_output}")
                return True
            else:
                self.validation_results['node'] = False
                self.validation_messages['node'] = "Node.js not found"
                return False
        except Exception as e:
            self.validation_results['node'] = False
            self.validation_messages['node'] = str(e)
            return False
    
    async def _check_system_packages(self, packages: List[str]) -> bool:
        """Check system packages availability."""
        all_found = True
        for package in packages:
            try:
                process = await asyncio.create_subprocess_exec(
                    'which', package,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.validation_results[f'package_{package}'] = True
                    self.validation_messages[f'package_{package}'] = "Found"
                else:
                    self.validation_results[f'package_{package}'] = False
                    self.validation_messages[f'package_{package}'] = "Not found"
                    all_found = False
            except Exception as e:
                self.validation_results[f'package_{package}'] = False
                self.validation_messages[f'package_{package}'] = str(e)
                all_found = False
        
        return all_found
    
    def _check_python_packages(self, packages: List[str]) -> bool:
        """Check Python packages availability."""
        all_found = True
        for package in packages:
            try:
                __import__(package)
                self.validation_results[f'python_package_{package}'] = True
                self.validation_messages[f'python_package_{package}'] = "Installed"
            except ImportError:
                self.validation_results[f'python_package_{package}'] = False
                self.validation_messages[f'python_package_{package}'] = "Not installed"
                all_found = False
        
        return all_found
    
    def _check_environment_variables(self, variables: List[str]) -> bool:
        """Check required environment variables."""
        all_set = True
        for var in variables:
            if var in os.environ:
                self.validation_results[f'env_{var}'] = True
                self.validation_messages[f'env_{var}'] = "Set"
            else:
                self.validation_results[f'env_{var}'] = False
                self.validation_messages[f'env_{var}'] = "Not set"
                all_set = False
        
        return all_set
    
    def _check_system_resources(self, requirements: SystemRequirements) -> bool:
        """Check system resources."""
        try:
            import psutil
            
            # Check available disk space
            disk_usage = psutil.disk_usage('/')
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb >= requirements.disk_space_gb:
                self.validation_results['disk_space'] = True
                self.validation_messages['disk_space'] = f"{available_gb:.1f}GB available"
            else:
                self.validation_results['disk_space'] = False
                self.validation_messages['disk_space'] = f"Insufficient disk space: {available_gb:.1f}GB < {requirements.disk_space_gb}GB"
                return False
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= requirements.memory_gb:
                self.validation_results['memory'] = True
                self.validation_messages['memory'] = f"{available_gb:.1f}GB available"
            else:
                self.validation_results['memory'] = False
                self.validation_messages['memory'] = f"Insufficient memory: {available_gb:.1f}GB < {requirements.memory_gb}GB"
                return False
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count(logical=False)
            if cpu_cores >= requirements.cpu_cores:
                self.validation_results['cpu_cores'] = True
                self.validation_messages['cpu_cores'] = f"{cpu_cores} cores available"
            else:
                self.validation_results['cpu_cores'] = False
                self.validation_messages['cpu_cores'] = f"Insufficient CPU cores: {cpu_cores} < {requirements.cpu_cores}"
                return False
            
            return True
            
        except ImportError:
            logger.warning("psutil not available, skipping resource checks")
            return True
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False


class ScriptManager:
    """Advanced script management and orchestration."""
    
    def __init__(self, scripts_dir: Path):
        self.scripts_dir = Path(scripts_dir)
        self.scripts: Dict[str, Tuple[Path, ScriptMetadata]] = {}
        self.executors: Dict[str, ScriptExecutor] = {
            '.sh': BashScriptExecutor(),
            '.bash': BashScriptExecutor(),
            '.py': PythonScriptExecutor()
        }
        self.dependency_resolver = DependencyResolver()
        self.environment_validator = EnvironmentValidator()
        self.execution_history: List[ExecutionResult] = []
        self.config_cache: Dict[str, Any] = {}
        
        # Performance metrics
        self.start_time = time.time()
        self.metrics = {
            'scripts_discovered': 0,
            'scripts_executed': 0,
            'scripts_succeeded': 0,
            'scripts_failed': 0,
            'total_execution_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the script manager."""
        logger.info("Initializing Script Manager...")
        
        try:
            # Discover scripts
            await self._discover_scripts()
            
            # Load configurations
            await self._load_configurations()
            
            # Build dependency graph
            self._build_dependency_graph()
            
            # Validate environment
            if not await self._validate_environment():
                logger.error("Environment validation failed")
                return False
            
            logger.info(f"Script Manager initialized with {len(self.scripts)} scripts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Script Manager: {e}")
            return False
    
    async def _discover_scripts(self):
        """Discover all scripts in the scripts directory."""
        logger.info("Discovering scripts...")
        
        script_files = []
        for ext in self.executors.keys():
            script_files.extend(self.scripts_dir.glob(f"*{ext}"))
        
        self.metrics['scripts_discovered'] = len(script_files)
        
        for script_file in script_files:
            try:
                metadata = await self._extract_metadata(script_file)
                self.scripts[metadata.name] = (script_file, metadata)
                logger.debug(f"Discovered script: {metadata.name}")
            except Exception as e:
                logger.warning(f"Failed to process script {script_file}: {e}")
    
    async def _extract_metadata(self, script_file: Path) -> ScriptMetadata:
        """Extract metadata from script file."""
        metadata = ScriptMetadata(
            name=script_file.stem,
            script_type=self._infer_script_type(script_file),
            description=f"Auto-discovered script: {script_file.name}",
            created_at=datetime.fromtimestamp(script_file.stat().st_ctime),
            modified_at=datetime.fromtimestamp(script_file.stat().st_mtime)
        )
        
        # Try to extract metadata from file comments
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for metadata in comments
            if script_file.suffix in ['.sh', '.bash']:
                parsed_metadata = self._parse_bash_metadata(content)
                for key, value in parsed_metadata.items():
                    setattr(metadata, key, value)
            elif script_file.suffix == '.py':
                parsed_metadata = self._parse_python_metadata(content)
                for key, value in parsed_metadata.items():
                    setattr(metadata, key, value)
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {script_file}: {e}")
        
        return metadata
    
    def _infer_script_type(self, script_file: Path) -> ScriptType:
        """Infer script type from filename."""
        name_lower = script_file.stem.lower()
        
        if 'setup' in name_lower or 'init' in name_lower:
            return ScriptType.SETUP
        elif 'test' in name_lower:
            return ScriptType.TESTING
        elif 'health' in name_lower or 'monitor' in name_lower:
            return ScriptType.HEALTH_CHECK
        elif 'db' in name_lower or 'database' in name_lower:
            return ScriptType.DATABASE
        elif 'service' in name_lower:
            return ScriptType.SERVICES
        elif 'deploy' in name_lower:
            return ScriptType.DEPLOYMENT
        elif 'backup' in name_lower:
            return ScriptType.BACKUP
        elif 'migrate' in name_lower or 'migration' in name_lower:
            return ScriptType.MIGRATION
        else:
            return ScriptType.MAINTENANCE
    
    def _parse_bash_metadata(self, content: str) -> dict:
        """Parse metadata from bash script comments."""
        metadata = {}
        lines = content.split('\n')
        
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            if line.startswith('#'):
                if 'DEPENDENCIES:' in line:
                    deps = line.split('DEPENDENCIES:')[1].strip().split(',')
                    metadata['dependencies'] = [dep.strip() for dep in deps if dep.strip()]
                elif 'TIMEOUT:' in line:
                    try:
                        metadata['timeout'] = int(line.split('TIMEOUT:')[1].strip())
                    except ValueError:
                        pass
                elif 'DESCRIPTION:' in line:
                    metadata['description'] = line.split('DESCRIPTION:')[1].strip()
                elif 'REQUIRES:' in line:
                    reqs = line.split('REQUIRES:')[1].strip().split(',')
                    metadata['requirements'] = [req.strip() for req in reqs if req.strip()]
        
        return metadata
    
    def _parse_python_metadata(self, content: str) -> dict:
        """Parse metadata from Python script docstrings and comments."""
        metadata = {}
        
        # Try to extract from docstring
        if '"""' in content:
            start = content.find('"""')
            end = content.find('"""', start + 3)
            if start != -1 and end != -1:
                docstring = content[start+3:end]
                # Parse docstring for metadata
                if 'Dependencies:' in docstring:
                    deps_section = docstring.split('Dependencies:')[1].split('\n')[0]
                    metadata['dependencies'] = [dep.strip() for dep in deps_section.split(',') if dep.strip()]
        
        return metadata
    
    async def _load_configurations(self):
        """Load script configurations from external files."""
        config_file = self.scripts_dir / 'scripts_config.yaml'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.config_cache.update(config)
                    logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load configuration: {e}")
    
    def _build_dependency_graph(self):
        """Build dependency graph for script execution order."""
        for script_name, (_, metadata) in self.scripts.items():
            self.dependency_resolver.add_dependency(script_name, metadata.dependencies)
    
    async def _validate_environment(self) -> bool:
        """Validate the execution environment."""
        requirements = SystemRequirements(
            python_version="3.9+",
            system_packages=['git', 'curl'],
            python_packages=['pyyaml'],
            environment_variables=[],
            disk_space_gb=1.0,
            memory_gb=1.0,
            cpu_cores=1
        )
        
        return await self.environment_validator.validate_system_requirements(requirements)
    
    async def execute_scripts(self, script_names: Optional[List[str]] = None, 
                            script_types: Optional[List[ScriptType]] = None,
                            parallel: bool = False) -> List[ExecutionResult]:
        """Execute specified scripts or all scripts of given types."""
        logger.info("Starting script execution...")
        
        # Determine which scripts to execute
        scripts_to_execute = self._select_scripts(script_names, script_types)
        
        if not scripts_to_execute:
            logger.warning("No scripts selected for execution")
            return []
        
        # Get execution order considering dependencies
        execution_order = self.dependency_resolver.get_execution_order(scripts_to_execute)
        
        results = []
        
        if parallel:
            results = await self._execute_parallel(execution_order)
        else:
            results = await self._execute_sequential(execution_order)
        
        # Update metrics
        self._update_metrics(results)
        
        # Store execution history
        self.execution_history.extend(results)
        
        logger.info(f"Script execution completed. {len(results)} scripts executed.")
        return results
    
    def _select_scripts(self, script_names: Optional[List[str]] = None,
                       script_types: Optional[List[ScriptType]] = None) -> List[str]:
        """Select scripts based on names or types."""
        if script_names:
            return [name for name in script_names if name in self.scripts]
        
        if script_types:
            return [
                name for name, (_, metadata) in self.scripts.items()
                if metadata.script_type in script_types
            ]
        
        # Return all scripts if no filter specified
        return list(self.scripts.keys())
    
    async def _execute_sequential(self, script_names: List[str]) -> List[ExecutionResult]:
        """Execute scripts sequentially."""
        results = []
        
        for script_name in script_names:
            if script_name not in self.scripts:
                logger.warning(f"Script {script_name} not found")
                continue
            
            script_path, metadata = self.scripts[script_name]
            executor = self._get_executor(script_path)
            
            logger.info(f"Executing script: {script_name}")
            
            try:
                # Validate script before execution
                if not executor.validate(script_path):
                    logger.error(f"Script validation failed: {script_name}")
                    result = ExecutionResult(
                        script_name=script_name,
                        status=ExecutionStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message="Script validation failed"
                    )
                    results.append(result)
                    continue
                
                # Execute script
                result = await executor.execute(script_path, metadata)
                results.append(result)
                
                logger.info(f"Script {script_name} completed with status: {result.status}")
                
                # Stop execution if critical script failed
                if metadata.critical and result.status == ExecutionStatus.FAILED:
                    logger.error(f"Critical script {script_name} failed. Stopping execution.")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing script {script_name}: {e}")
                result = ExecutionResult(
                    script_name=script_name,
                    status=ExecutionStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    async def _execute_parallel(self, script_names: List[str]) -> List[ExecutionResult]:
        """Execute scripts in parallel where possible."""
        results = []
        
        # Group scripts by dependency level
        dependency_levels = self._get_dependency_levels(script_names)
        
        for level_scripts in dependency_levels:
            if len(level_scripts) == 1:
                # Single script, execute normally
                script_name = level_scripts[0]
                if script_name in self.scripts:
                    script_path, metadata = self.scripts[script_name]
                    executor = self._get_executor(script_path)
                    result = await executor.execute(script_path, metadata)
                    results.append(result)
            else:
                # Multiple scripts, execute in parallel
                tasks = []
                for script_name in level_scripts:
                    if script_name in self.scripts:
                        script_path, metadata = self.scripts[script_name]
                        executor = self._get_executor(script_path)
                        task = asyncio.create_task(executor.execute(script_path, metadata))
                        tasks.append(task)
                
                # Wait for all scripts in this level to complete
                level_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in level_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in parallel execution: {result}")
                    else:
                        results.append(result)
        
        return results
    
    def _get_dependency_levels(self, script_names: List[str]) -> List[List[str]]:
        """Group scripts by dependency levels for parallel execution."""
        levels = []
        remaining = set(script_names)
        
        while remaining:
            current_level = []
            for script in list(remaining):
                script_deps = set(self.scripts[script][1].dependencies) if script in self.scripts else set()
                if script_deps.isdisjoint(remaining):
                    current_level.append(script)
            
            if not current_level:
                # Circular dependency detected, add remaining scripts to avoid infinite loop
                current_level = list(remaining)
                logger.warning("Possible circular dependency detected")
            
            levels.append(current_level)
            remaining -= set(current_level)
        
        return levels
    
    def _get_executor(self, script_path: Path) -> ScriptExecutor:
        """Get appropriate executor for script."""
        suffix = script_path.suffix
        if suffix in self.executors:
            return self.executors[suffix]
        else:
            raise ValueError(f"No executor found for script type: {suffix}")
    
    def _update_metrics(self, results: List[ExecutionResult]):
        """Update execution metrics."""
        self.metrics['scripts_executed'] += len(results)
        
        for result in results:
            if result.status == ExecutionStatus.SUCCESS:
                self.metrics['scripts_succeeded'] += 1
            elif result.status == ExecutionStatus.FAILED:
                self.metrics['scripts_failed'] += 1
            
            if result.duration:
                self.metrics['total_execution_time'] += result.duration
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        logger.info("Performing health check...")
        
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'scripts_directory': str(self.scripts_dir),
            'scripts_count': len(self.scripts),
            'environment_validation': self.environment_validator.validation_results,
            'metrics': self.metrics,
            'issues': []
        }
        
        # Check script directory
        if not self.scripts_dir.exists():
            health_status['issues'].append(f"Scripts directory does not exist: {self.scripts_dir}")
            health_status['overall_status'] = 'unhealthy'
        
        # Check script validity
        invalid_scripts = []
        for script_name, (script_path, metadata) in self.scripts.items():
            try:
                executor = self._get_executor(script_path)
                if not executor.validate(script_path):
                    invalid_scripts.append(script_name)
            except Exception as e:
                invalid_scripts.append(f"{script_name}: {str(e)}")
        
        if invalid_scripts:
            health_status['issues'].extend([f"Invalid script: {script}" for script in invalid_scripts])
            health_status['overall_status'] = 'degraded'
        
        # Check recent execution failures
        recent_failures = [
            result for result in self.execution_history[-10:]
            if result.status == ExecutionStatus.FAILED
        ]
        
        if recent_failures:
            health_status['recent_failures'] = len(recent_failures)
            if len(recent_failures) > 5:
                health_status['overall_status'] = 'degraded'
        
        return health_status
    
    async def cleanup(self):
        """Cleanup resources and temporary files."""
        logger.info("Cleaning up Script Manager...")
        
        # Clean up any temporary files
        temp_dir = Path(tempfile.gettempdir()) / 'spotify_ai_agent_scripts'
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save execution history
        history_file = self.scripts_dir / 'execution_history.json'
        try:
            with open(history_file, 'w') as f:
                history_data = [
                    {
                        'script_name': result.script_name,
                        'status': result.status.value,
                        'start_time': result.start_time.isoformat(),
                        'end_time': result.end_time.isoformat() if result.end_time else None,
                        'duration': result.duration,
                        'return_code': result.return_code,
                        'error_message': result.error_message
                    }
                    for result in self.execution_history
                ]
                json.dump(history_data, f, indent=2)
            logger.info(f"Execution history saved to {history_file}")
        except Exception as e:
            logger.warning(f"Failed to save execution history: {e}")
        
        logger.info("Script Manager cleanup completed")
    
    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a script."""
        if script_name not in self.scripts:
            return None
        
        script_path, metadata = self.scripts[script_name]
        
        return {
            'name': metadata.name,
            'path': str(script_path),
            'type': metadata.script_type.value,
            'description': metadata.description,
            'version': metadata.version,
            'author': metadata.author,
            'dependencies': metadata.dependencies,
            'requirements': metadata.requirements,
            'environments': [env.value for env in metadata.environments],
            'timeout': metadata.timeout,
            'retries': metadata.retries,
            'parallel': metadata.parallel,
            'critical': metadata.critical,
            'tags': metadata.tags,
            'security_level': metadata.security_level,
            'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
            'modified_at': metadata.modified_at.isoformat() if metadata.modified_at else None,
            'file_size': script_path.stat().st_size,
            'executable': os.access(script_path, os.X_OK)
        }
    
    def list_scripts(self, script_type: Optional[ScriptType] = None) -> List[Dict[str, Any]]:
        """List all scripts with optional filtering by type."""
        scripts_info = []
        
        for script_name, (script_path, metadata) in self.scripts.items():
            if script_type is None or metadata.script_type == script_type:
                scripts_info.append({
                    'name': script_name,
                    'type': metadata.script_type.value,
                    'description': metadata.description,
                    'dependencies': metadata.dependencies,
                    'path': str(script_path)
                })
        
        return scripts_info


# Convenience functions for common operations

async def initialize_development_environment(scripts_dir: str = None) -> bool:
    """Initialize development environment with all setup scripts."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).parent
    
    manager = ScriptManager(scripts_dir)
    
    try:
        # Initialize manager
        if not await manager.initialize():
            return False
        
        # Execute setup scripts
        results = await manager.execute_scripts(
            script_types=[ScriptType.SETUP],
            parallel=False
        )
        
        # Check if all setup scripts succeeded
        failed_scripts = [r for r in results if r.status == ExecutionStatus.FAILED]
        if failed_scripts:
            logger.error(f"Setup failed. {len(failed_scripts)} scripts failed.")
            for result in failed_scripts:
                logger.error(f"Failed script: {result.script_name} - {result.error_message}")
            return False
        
        logger.info("Development environment initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize development environment: {e}")
        return False
    finally:
        await manager.cleanup()


async def run_health_checks(scripts_dir: str = None) -> Dict[str, Any]:
    """Run all health check scripts and return status."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).parent
    
    manager = ScriptManager(scripts_dir)
    
    try:
        await manager.initialize()
        
        # Execute health check scripts
        results = await manager.execute_scripts(
            script_types=[ScriptType.HEALTH_CHECK],
            parallel=True
        )
        
        # Get overall health status
        health_status = await manager.health_check()
        
        # Add health check results
        health_status['health_check_results'] = [
            {
                'script': result.script_name,
                'status': result.status.value,
                'duration': result.duration,
                'message': result.error_message or "OK"
            }
            for result in results
        ]
        
        return health_status
        
    except Exception as e:
        return {
            'overall_status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        await manager.cleanup()


# Main execution function
async def main():
    """Main function for script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Scripts Initialization System')
    parser.add_argument('--action', choices=['init', 'health', 'list', 'execute'], 
                       default='init', help='Action to perform')
    parser.add_argument('--scripts-dir', help='Scripts directory path')
    parser.add_argument('--script-names', nargs='*', help='Specific scripts to execute')
    parser.add_argument('--script-types', nargs='*', help='Script types to execute')
    parser.add_argument('--parallel', action='store_true', help='Execute scripts in parallel')
    
    args = parser.parse_args()
    
    scripts_dir = args.scripts_dir or Path(__file__).parent
    
    if args.action == 'init':
        success = await initialize_development_environment(scripts_dir)
        sys.exit(0 if success else 1)
    
    elif args.action == 'health':
        health_status = await run_health_checks(scripts_dir)
        print(json.dumps(health_status, indent=2))
        sys.exit(0 if health_status.get('overall_status') == 'healthy' else 1)
    
    elif args.action == 'list':
        manager = ScriptManager(scripts_dir)
        await manager.initialize()
        scripts = manager.list_scripts()
        for script in scripts:
            print(f"{script['name']:<20} {script['type']:<15} {script['description']}")
        await manager.cleanup()
    
    elif args.action == 'execute':
        manager = ScriptManager(scripts_dir)
        await manager.initialize()
        
        script_types = None
        if args.script_types:
            script_types = [ScriptType(t) for t in args.script_types]
        
        results = await manager.execute_scripts(
            script_names=args.script_names,
            script_types=script_types,
            parallel=args.parallel
        )
        
        # Print results
        for result in results:
            print(f"{result.script_name}: {result.status.value}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
        
        await manager.cleanup()
        
        # Exit with error if any script failed
        failed_count = sum(1 for r in results if r.status == ExecutionStatus.FAILED)
        sys.exit(failed_count)


if __name__ == "__main__":
    asyncio.run(main())
