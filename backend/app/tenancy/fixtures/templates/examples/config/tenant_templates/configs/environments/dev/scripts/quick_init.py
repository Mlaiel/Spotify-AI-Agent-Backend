#!/usr/bin/env python3
"""
Quick Development Environment Initialization Script
==================================================

Ultra-fast initialization script for the Spotify AI Agent development environment.
This script demonstrates the advanced script management system capabilities.

Developed by: Fahed Mlaiel
Expert Team: Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
            DBA & Data Engineer, Backend Security Specialist, Microservices Architect

Dependencies: setup_dev
Timeout: 60
Description: Quick initialization with validation and health checks
"""

import sys
import time
import json
import subprocess
from pathlib import Path

def log_info(message):
    """Log information message."""
    print(f"[INFO] {message}")

def log_success(message):
    """Log success message."""
    print(f"[SUCCESS] {message}")

def log_error(message):
    """Log error message."""
    print(f"[ERROR] {message}")

def check_python_environment():
    """Check Python environment setup."""
    log_info("Checking Python environment...")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        log_success(f"Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    else:
        log_error(f"Python version too old: {version.major}.{version.minor}.{version.micro}")
        return False

def check_system_tools():
    """Check required system tools."""
    log_info("Checking system tools...")
    
    tools = ['git', 'curl', 'python3']
    all_found = True
    
    for tool in tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                log_success(f"{tool} found: {result.stdout.strip()}")
            else:
                log_error(f"{tool} not found")
                all_found = False
        except Exception as e:
            log_error(f"Error checking {tool}: {e}")
            all_found = False
    
    return all_found

def create_demo_structure():
    """Create demo project structure."""
    log_info("Creating demo project structure...")
    
    try:
        # Create demo directories
        dirs = [
            'tmp/demo_project',
            'tmp/demo_project/src',
            'tmp/demo_project/tests',
            'tmp/demo_project/docs',
            'tmp/demo_project/config'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create demo files
        files = {
            'tmp/demo_project/README.md': '# Demo Project\n\nThis is a demo project created by the initialization script.\n',
            'tmp/demo_project/src/__init__.py': '"""Demo package."""\n\n__version__ = "1.0.0"\n',
            'tmp/demo_project/config/settings.json': json.dumps({
                'environment': 'development',
                'debug': True,
                'created_by': 'Fahed Mlaiel Expert Team'
            }, indent=2)
        }
        
        for file_path, content in files.items():
            with open(file_path, 'w') as f:
                f.write(content)
        
        log_success("Demo project structure created")
        return True
        
    except Exception as e:
        log_error(f"Failed to create demo structure: {e}")
        return False

def run_health_checks():
    """Run basic health checks."""
    log_info("Running health checks...")
    
    checks = []
    
    # Check 1: File system access
    try:
        test_file = Path('tmp/health_check.txt')
        test_file.write_text('Health check test')
        test_file.unlink()
        checks.append(('filesystem', True, 'File system access OK'))
    except Exception as e:
        checks.append(('filesystem', False, f'File system error: {e}'))
    
    # Check 2: Python imports
    try:
        import json
        import sys
        import pathlib
        checks.append(('python_imports', True, 'Standard library imports OK'))
    except Exception as e:
        checks.append(('python_imports', False, f'Import error: {e}'))
    
    # Check 3: Script execution permissions
    try:
        script_path = Path(__file__)
        if script_path.stat().st_mode & 0o111:
            checks.append(('permissions', True, 'Script execution permissions OK'))
        else:
            checks.append(('permissions', False, 'Script not executable'))
    except Exception as e:
        checks.append(('permissions', False, f'Permission check error: {e}'))
    
    # Report results
    all_passed = True
    for check_name, passed, message in checks:
        if passed:
            log_success(f"Health check '{check_name}': {message}")
        else:
            log_error(f"Health check '{check_name}': {message}")
            all_passed = False
    
    return all_passed

def main():
    """Main initialization function."""
    start_time = time.time()
    
    log_info("=== Spotify AI Agent Quick Initialization ===")
    log_info("Developed by: Fahed Mlaiel Expert Team")
    log_info("Starting quick development environment setup...")
    
    success = True
    
    # Step 1: Check Python environment
    if not check_python_environment():
        success = False
    
    # Step 2: Check system tools
    if not check_system_tools():
        success = False
    
    # Step 3: Create demo structure
    if not create_demo_structure():
        success = False
    
    # Step 4: Run health checks
    if not run_health_checks():
        success = False
    
    # Final report
    end_time = time.time()
    duration = end_time - start_time
    
    log_info(f"Initialization completed in {duration:.2f} seconds")
    
    if success:
        log_success("=== Quick Initialization Successful ===")
        log_info("Environment is ready for development!")
        
        # Create success indicator file
        success_file = Path('tmp/initialization_success.json')
        success_data = {
            'status': 'success',
            'timestamp': time.time(),
            'duration': duration,
            'script': 'quick_init.py',
            'author': 'Fahed Mlaiel Expert Team'
        }
        
        with open(success_file, 'w') as f:
            json.dump(success_data, f, indent=2)
        
        return 0
    else:
        log_error("=== Quick Initialization Failed ===")
        log_error("Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
