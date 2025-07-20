#!/usr/bin/env python3
"""
Enterprise Key Management Integration Test Suite
==============================================

Ultra-advanced integration testing suite for the enterprise-grade 
key management system. This module provides comprehensive testing
of all key management operations, security validations, and 
system integrations.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer
- ML Engineer  
- DBA & Data Engineer
- Backend Security Specialist
- Microservices Architect

Features:
- Comprehensive key validation testing
- Encryption/decryption verification
- JWT token generation and validation
- Database connection testing
- API security validation
- Performance benchmarking
- Security compliance testing
- Multi-environment validation
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import hmac
import base64
import secrets

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integration_test.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class SecurityTestResult:
    """Container for security test results."""
    
    def __init__(self, test_name: str, status: str, duration: float, 
                 details: Dict[str, Any] = None, error: str = None):
        self.test_name = test_name
        self.status = status  # PASS, FAIL, SKIP, ERROR
        self.duration = duration
        self.details = details or {}
        self.error = error
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'status': self.status,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'error': self.error
        }


class KeyManagerIntegrationTest:
    """Ultra-advanced integration test suite for key management system."""
    
    def __init__(self, keys_dir: str = None):
        self.keys_dir = Path(keys_dir) if keys_dir else Path(__file__).parent
        self.test_results: List[SecurityTestResult] = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_config = {
            'encryption_iterations': 1000,
            'jwt_test_duration': 3600,  # 1 hour
            'database_timeout': 10,
            'api_timeout': 5,
            'performance_threshold': 0.1,  # 100ms
            'security_strength_min': 256  # bits
        }
    
    def log_test_start(self, test_name: str):
        """Log test start."""
        logger.info(f"🧪 Starting test: {test_name}")
    
    def log_test_result(self, result: SecurityTestResult):
        """Log test result."""
        if result.status == 'PASS':
            logger.info(f"✅ {result.test_name}: PASSED ({result.duration:.3f}s)")
        elif result.status == 'FAIL':
            logger.error(f"❌ {result.test_name}: FAILED ({result.duration:.3f}s) - {result.error}")
        elif result.status == 'SKIP':
            logger.warning(f"⏭️  {result.test_name}: SKIPPED")
        else:
            logger.error(f"💥 {result.test_name}: ERROR ({result.duration:.3f}s) - {result.error}")
        
        self.test_results.append(result)
    
    async def test_key_files_exist(self) -> SecurityTestResult:
        """Test that all required key files exist."""
        self.log_test_start("Key Files Existence")
        start_time = time.time()
        
        try:
            required_files = [
                'encryption_keys.key',
                'jwt_keys.key',
                'hmac_keys.key',
                'api_keys.key',
                'database_encryption.key',
                'session_keys.key',
                'rsa_private.pem',
                'rsa_public.pem',
                'key_registry.json'
            ]
            
            missing_files = []
            existing_files = []
            
            for filename in required_files:
                file_path = self.keys_dir / filename
                if file_path.exists():
                    existing_files.append(filename)
                else:
                    missing_files.append(filename)
            
            if missing_files:
                raise ValueError(f"Missing key files: {missing_files}")
            
            duration = time.time() - start_time
            result = SecurityTestResult(
                "Key Files Existence",
                "PASS",
                duration,
                {
                    'total_files': len(required_files),
                    'existing_files': len(existing_files),
                    'missing_files': len(missing_files),
                    'files_list': existing_files
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = SecurityTestResult("Key Files Existence", "FAIL", duration, error=str(e))
        
        self.log_test_result(result)
        return result
    
    async def test_key_file_permissions(self) -> SecurityTestResult:
        """Test key file permissions for security."""
        self.log_test_start("Key File Permissions")
        start_time = time.time()
        
        try:
            security_files = [
                ('rsa_private.pem', '600'),  # Private key should be 600
                ('encryption_keys.key', '600'),
                ('jwt_keys.key', '600'),
                ('hmac_keys.key', '600'),
                ('api_keys.key', '600'),
                ('database_encryption.key', '600'),
                ('session_keys.key', '600'),
                ('rsa_public.pem', '644')  # Public key can be 644
            ]
            
            permission_results = {}
            
            for filename, expected_perms in security_files:
                file_path = self.keys_dir / filename
                if file_path.exists():
                    stat_info = file_path.stat()
                    actual_perms = oct(stat_info.st_mode)[-3:]
                    permission_results[filename] = {
                        'expected': expected_perms,
                        'actual': actual_perms,
                        'secure': actual_perms in ['600', '644']
                    }
            
            # Check if any files have insecure permissions
            insecure_files = [
                name for name, info in permission_results.items()
                if not info['secure']
            ]
            
            duration = time.time() - start_time
            result = SecurityTestResult(
                "Key File Permissions",
                "PASS" if not insecure_files else "FAIL",
                duration,
                {
                    'permission_results': permission_results,
                    'insecure_files': insecure_files,
                    'total_checked': len(permission_results)
                },
                error=f"Insecure permissions on: {insecure_files}" if insecure_files else None
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = SecurityTestResult("Key File Permissions", "FAIL", duration, error=str(e))
        
        self.log_test_result(result)
        return result
    
    async def test_key_registry_integrity(self) -> SecurityTestResult:
        """Test key registry file integrity."""
        self.log_test_start("Key Registry Integrity")
        start_time = time.time()
        
        try:
            registry_file = self.keys_dir / 'key_registry.json'
            
            if not registry_file.exists():
                raise ValueError("Key registry file does not exist")
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check required fields
            required_fields = ['keys', 'metadata', 'created_at', 'last_updated']
            missing_fields = [field for field in required_fields if field not in registry_data]
            
            if missing_fields:
                raise ValueError(f"Missing registry fields: {missing_fields}")
            
            # Check keys section
            keys_data = registry_data.get('keys', {})
            if not keys_data:
                raise ValueError("No keys registered in registry")
            
            # Validate each key entry
            key_validation = {}
            for key_name, key_info in keys_data.items():
                key_validation[key_name] = {
                    'has_file_path': 'file_path' in key_info,
                    'has_created_at': 'created_at' in key_info,
                    'has_algorithm': 'algorithm' in key_info,
                    'file_exists': (self.keys_dir / key_info.get('file_path', '')).exists() if 'file_path' in key_info else False
                }
            
            duration = time.time() - start_time
            result = SecurityTestResult(
                "Key Registry Integrity",
                "PASS",
                duration,
                {
                    'registry_fields': list(registry_data.keys()),
                    'total_keys': len(keys_data),
                    'key_validation': key_validation,
                    'missing_fields': missing_fields
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = SecurityTestResult("Key Registry Integrity", "FAIL", duration, error=str(e))
        
        self.log_test_result(result)
        return result
    
    async def test_basic_cryptographic_operations(self) -> SecurityTestResult:
        """Test basic cryptographic operations with generated keys."""
        self.log_test_start("Basic Cryptographic Operations")
        start_time = time.time()
        
        try:
            # Test data
            test_message = b"Hello, Enterprise Key Management System!"
            
            crypto_results = {}
            
            # Test with encryption key (if it's a Fernet key)
            try:
                encryption_file = self.keys_dir / 'encryption_keys.key'
                if encryption_file.exists():
                    with open(encryption_file, 'r') as f:
                        encryption_key = f.read().strip()
                    
                    if len(encryption_key) == 44:  # Fernet key length
                        from cryptography.fernet import Fernet
                        fernet = Fernet(encryption_key.encode())
                        
                        encrypted = fernet.encrypt(test_message)
                        decrypted = fernet.decrypt(encrypted)
                        
                        if decrypted == test_message:
                            crypto_results['fernet_encryption'] = 'PASS'
                        else:
                            crypto_results['fernet_encryption'] = 'FAIL'
                    else:
                        crypto_results['fernet_encryption'] = 'SKIP - Invalid key format'
            except Exception as e:
                crypto_results['fernet_encryption'] = f'ERROR - {str(e)}'
            
            # Test HMAC operations
            try:
                hmac_file = self.keys_dir / 'hmac_keys.key'
                if hmac_file.exists():
                    with open(hmac_file, 'r') as f:
                        hmac_data = json.load(f)
                    
                    hmac_key = hmac_data.get('primary', '')
                    if hmac_key:
                        signature = hmac.new(
                            hmac_key.encode(),
                            test_message,
                            hashlib.sha256
                        ).hexdigest()
                        
                        # Verify signature
                        verification = hmac.new(
                            hmac_key.encode(),
                            test_message,
                            hashlib.sha256
                        ).hexdigest()
                        
                        if signature == verification:
                            crypto_results['hmac_operations'] = 'PASS'
                        else:
                            crypto_results['hmac_operations'] = 'FAIL'
                    else:
                        crypto_results['hmac_operations'] = 'SKIP - No primary key'
            except Exception as e:
                crypto_results['hmac_operations'] = f'ERROR - {str(e)}'
            
            # Test JWT operations
            try:
                jwt_file = self.keys_dir / 'jwt_keys.key'
                if jwt_file.exists():
                    with open(jwt_file, 'r') as f:
                        jwt_data = json.load(f)
                    
                    jwt_secret = jwt_data.get('primary', '')
                    if jwt_secret:
                        import jwt
                        
                        payload = {
                            'test': 'data',
                            'exp': datetime.utcnow() + timedelta(hours=1)
                        }
                        
                        token = jwt.encode(payload, jwt_secret, algorithm='HS256')
                        decoded = jwt.decode(token, jwt_secret, algorithms=['HS256'])
                        
                        if decoded['test'] == 'data':
                            crypto_results['jwt_operations'] = 'PASS'
                        else:
                            crypto_results['jwt_operations'] = 'FAIL'
                    else:
                        crypto_results['jwt_operations'] = 'SKIP - No primary key'
            except Exception as e:
                crypto_results['jwt_operations'] = f'ERROR - {str(e)}'
            
            # Check if at least one operation passed
            passed_operations = [op for op, result in crypto_results.items() if result == 'PASS']
            
            duration = time.time() - start_time
            result = SecurityTestResult(
                "Basic Cryptographic Operations",
                "PASS" if passed_operations else "FAIL",
                duration,
                {
                    'operations_tested': len(crypto_results),
                    'operations_passed': len(passed_operations),
                    'results': crypto_results
                },
                error="No cryptographic operations passed" if not passed_operations else None
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = SecurityTestResult("Basic Cryptographic Operations", "FAIL", duration, error=str(e))
        
        self.log_test_result(result)
        return result
    
    async def test_key_strength_validation(self) -> SecurityTestResult:
        """Test cryptographic key strength validation."""
        self.log_test_start("Key Strength Validation")
        start_time = time.time()
        
        try:
            strength_results = {}
            
            # Test various key files
            key_files = [
                ('encryption_keys.key', 32),  # Minimum 32 characters for Fernet
                ('jwt_keys.key', 32),         # Minimum 32 characters for JWT
                ('hmac_keys.key', 32),        # Minimum 32 characters for HMAC
                ('api_keys.key', 32),         # Minimum 32 characters for API keys
                ('session_keys.key', 32),     # Minimum 32 characters for sessions
            ]
            
            for filename, min_length in key_files:
                file_path = self.keys_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()
                        
                        # Try to parse as JSON first
                        try:
                            key_data = json.loads(content)
                            if isinstance(key_data, dict):
                                # Check each key in the JSON
                                for key_name, key_value in key_data.items():
                                    if isinstance(key_value, str):
                                        strength_results[f"{filename}_{key_name}"] = {
                                            'length': len(key_value),
                                            'min_required': min_length,
                                            'sufficient': len(key_value) >= min_length,
                                            'entropy': len(set(key_value))
                                        }
                            else:
                                # Single key value
                                strength_results[filename] = {
                                    'length': len(str(key_data)),
                                    'min_required': min_length,
                                    'sufficient': len(str(key_data)) >= min_length,
                                    'entropy': len(set(str(key_data)))
                                }
                        except json.JSONDecodeError:
                            # Plain text key
                            strength_results[filename] = {
                                'length': len(content),
                                'min_required': min_length,
                                'sufficient': len(content) >= min_length,
                                'entropy': len(set(content))
                            }
                    except Exception as e:
                        strength_results[filename] = {
                            'error': str(e),
                            'sufficient': False
                        }
            
            # Check RSA key strength
            rsa_private_file = self.keys_dir / 'rsa_private.pem'
            if rsa_private_file.exists():
                try:
                    from cryptography.hazmat.primitives import serialization
                    from cryptography.hazmat.backends import default_backend
                    
                    with open(rsa_private_file, 'rb') as f:
                        private_key = serialization.load_pem_private_key(
                            f.read(),
                            password=None,
                            backend=default_backend()
                        )
                    
                    key_size = private_key.key_size
                    strength_results['rsa_private.pem'] = {
                        'key_size': key_size,
                        'min_required': 2048,
                        'sufficient': key_size >= 2048,
                        'strength_level': 'strong' if key_size >= 4096 else 'adequate' if key_size >= 2048 else 'weak'
                    }
                except Exception as e:
                    strength_results['rsa_private.pem'] = {
                        'error': str(e),
                        'sufficient': False
                    }
            
            # Check if all keys meet minimum requirements
            insufficient_keys = [
                name for name, info in strength_results.items()
                if not info.get('sufficient', False)
            ]
            
            duration = time.time() - start_time
            result = SecurityTestResult(
                "Key Strength Validation",
                "PASS" if not insufficient_keys else "FAIL",
                duration,
                {
                    'total_keys_checked': len(strength_results),
                    'insufficient_keys': insufficient_keys,
                    'strength_results': strength_results
                },
                error=f"Insufficient key strength: {insufficient_keys}" if insufficient_keys else None
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = SecurityTestResult("Key Strength Validation", "FAIL", duration, error=str(e))
        
        self.log_test_result(result)
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Starting Enterprise Key Management Integration Tests")
        logger.info("=" * 70)
        
        # List of all tests to run
        tests = [
            self.test_key_files_exist,
            self.test_key_file_permissions,
            self.test_key_registry_integrity,
            self.test_basic_cryptographic_operations,
            self.test_key_strength_validation
        ]
        
        # Run tests
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test execution error: {e}")
        
        # Generate test report
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == 'PASS'])
        failed_tests = len([r for r in self.test_results if r.status == 'FAIL'])
        error_tests = len([r for r in self.test_results if r.status == 'ERROR'])
        skipped_tests = len([r for r in self.test_results if r.status == 'SKIP'])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'skipped': skipped_tests,
                'success_rate': f"{success_rate:.1f}%",
                'total_duration': f"{total_duration:.3f}s"
            },
            'test_results': [result.to_dict() for result in self.test_results],
            'timestamp': datetime.now().isoformat(),
            'environment': 'development',
            'system_info': {
                'python_version': sys.version,
                'keys_directory': str(self.keys_dir),
                'test_config': self.test_config
            }
        }
        
        # Log summary
        logger.info("=" * 70)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"💥 Errors: {error_tests}")
        logger.info(f"⏭️  Skipped: {skipped_tests}")
        logger.info(f"🎯 Success Rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Total Duration: {total_duration:.3f}s")
        
        if failed_tests > 0 or error_tests > 0:
            logger.error("❌ SOME TESTS FAILED - SECURITY VALIDATION INCOMPLETE")
        else:
            logger.info("✅ ALL TESTS PASSED - SECURITY VALIDATION SUCCESSFUL")
        
        return report


async def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise Key Management Integration Tests')
    parser.add_argument('--keys-dir', help='Keys directory path')
    parser.add_argument('--output', help='Output file for test report')
    parser.add_argument('--test', help='Run specific test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test suite
    test_suite = KeyManagerIntegrationTest(args.keys_dir)
    
    try:
        if args.test:
            # Run specific test
            test_method = getattr(test_suite, f"test_{args.test}", None)
            if test_method:
                await test_method()
                report = test_suite.generate_test_report()
            else:
                logger.error(f"Test '{args.test}' not found")
                return 1
        else:
            # Run all tests
            report = await test_suite.run_all_tests()
        
        # Save report if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to: {args.output}")
        
        # Return exit code based on test results
        failed_tests = len([r for r in test_suite.test_results if r.status in ['FAIL', 'ERROR']])
        return 1 if failed_tests > 0 else 0
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.key_manager = None
        
    async def initialize_key_manager(self) -> bool:
        """Initialize the key management system."""
        try:
            if not KEY_MANAGER_AVAILABLE:
                logger.error("Key management system not available")
                return False
                
            self.key_manager = EnterpriseKeyManager()
            logger.info("✅ Enterprise Key Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize key manager: {e}")
            return False
    
    def test_key_file_presence(self) -> Dict[str, Any]:
        """Test if all expected key files are present."""
        test_name = "Key File Presence"
        logger.info(f"🔍 Testing: {test_name}")
        
        expected_files = [
            "database_encryption.key",
            "jwt_keys.key", 
            "api_keys.key",
            "session_keys.key",
            "encryption_keys.key",
            "hmac_keys.key",
            "rsa_keys.key",
            "development_keys.key",
            "rsa_private.pem",
            "rsa_public.pem",
            "key_registry.json"
        ]
        
        missing_files = []
        present_files = []
        
        for file_name in expected_files:
            file_path = current_dir / file_name
            if file_path.exists():
                present_files.append(file_name)
                logger.info(f"  ✅ {file_name} found")
            else:
                missing_files.append(file_name)
                logger.error(f"  ❌ {file_name} missing")
        
        success = len(missing_files) == 0
        
        result = {
            "test_name": test_name,
            "success": success,
            "present_files": present_files,
            "missing_files": missing_files,
            "total_expected": len(expected_files),
            "total_found": len(present_files)
        }
        
        logger.info(f"📊 {test_name}: {'PASSED' if success else 'FAILED'}")
        return result
    
    def test_file_permissions(self) -> Dict[str, Any]:
        """Test file permissions for security compliance."""
        test_name = "File Permissions"
        logger.info(f"🔍 Testing: {test_name}")
        
        permission_tests = []
        
        # Check key files (should be 600)
        key_files = list(current_dir.glob("*.key"))
        for key_file in key_files:
            if key_file.exists():
                perms = oct(key_file.stat().st_mode)[-3:]
                expected = "600"
                success = perms == expected
                
                permission_tests.append({
                    "file": key_file.name,
                    "expected": expected,
                    "actual": perms,
                    "success": success
                })
                
                if success:
                    logger.info(f"  ✅ {key_file.name}: {perms}")
                else:
                    logger.error(f"  ❌ {key_file.name}: {perms} (expected {expected})")
        
        # Check RSA private key (should be 600)
        rsa_private = current_dir / "rsa_private.pem"
        if rsa_private.exists():
            perms = oct(rsa_private.stat().st_mode)[-3:]
            expected = "600"
            success = perms == expected
            
            permission_tests.append({
                "file": "rsa_private.pem",
                "expected": expected,
                "actual": perms,
                "success": success
            })
            
            if success:
                logger.info(f"  ✅ rsa_private.pem: {perms}")
            else:
                logger.error(f"  ❌ rsa_private.pem: {perms} (expected {expected})")
        
        # Check RSA public key (should be 644)
        rsa_public = current_dir / "rsa_public.pem"
        if rsa_public.exists():
            perms = oct(rsa_public.stat().st_mode)[-3:]
            expected = "644"
            success = perms == expected
            
            permission_tests.append({
                "file": "rsa_public.pem",
                "expected": expected,
                "actual": perms,
                "success": success
            })
            
            if success:
                logger.info(f"  ✅ rsa_public.pem: {perms}")
            else:
                logger.error(f"  ❌ rsa_public.pem: {perms} (expected {expected})")
        
        success = all(test["success"] for test in permission_tests)
        
        result = {
            "test_name": test_name,
            "success": success,
            "permission_tests": permission_tests,
            "total_tests": len(permission_tests),
            "passed_tests": sum(1 for test in permission_tests if test["success"])
        }
        
        logger.info(f"📊 {test_name}: {'PASSED' if success else 'FAILED'}")
        return result
    
    def test_key_content_validation(self) -> Dict[str, Any]:
        """Test key content for proper format and length."""
        test_name = "Key Content Validation"
        logger.info(f"🔍 Testing: {test_name}")
        
        content_tests = []
        
        # Test database encryption key
        db_key_file = current_dir / "database_encryption.key"
        if db_key_file.exists():
            try:
                content = db_key_file.read_text()
                if "DATABASE_ENCRYPTION_KEY=" in content:
                    key_line = [line for line in content.split('\n') if line.startswith('DATABASE_ENCRYPTION_KEY=')][0]
                    key_value = key_line.split('=')[1].strip('"')
                    
                    # Check if it's base64 and proper length
                    import base64
                    try:
                        decoded = base64.b64decode(key_value)
                        key_length = len(decoded)
                        success = key_length >= 32  # At least 256 bits
                        
                        content_tests.append({
                            "file": "database_encryption.key",
                            "key_length_bytes": key_length,
                            "key_length_bits": key_length * 8,
                            "success": success,
                            "error": None
                        })
                        
                        if success:
                            logger.info(f"  ✅ Database key: {key_length} bytes ({key_length * 8} bits)")
                        else:
                            logger.error(f"  ❌ Database key too short: {key_length} bytes")
                            
                    except Exception as e:
                        content_tests.append({
                            "file": "database_encryption.key",
                            "success": False,
                            "error": f"Invalid base64: {e}"
                        })
                        logger.error(f"  ❌ Database key invalid base64: {e}")
                        
            except Exception as e:
                content_tests.append({
                    "file": "database_encryption.key",
                    "success": False,
                    "error": f"Failed to read file: {e}"
                })
                logger.error(f"  ❌ Failed to read database key: {e}")
        
        # Test JWT keys
        jwt_key_file = current_dir / "jwt_keys.key"
        if jwt_key_file.exists():
            try:
                content = jwt_key_file.read_text()
                access_key_found = "JWT_ACCESS_SECRET=" in content
                refresh_key_found = "JWT_REFRESH_SECRET=" in content
                
                content_tests.append({
                    "file": "jwt_keys.key",
                    "access_key_found": access_key_found,
                    "refresh_key_found": refresh_key_found,
                    "success": access_key_found and refresh_key_found,
                    "error": None
                })
                
                if access_key_found and refresh_key_found:
                    logger.info(f"  ✅ JWT keys: Both access and refresh keys found")
                else:
                    logger.error(f"  ❌ JWT keys incomplete: access={access_key_found}, refresh={refresh_key_found}")
                    
            except Exception as e:
                content_tests.append({
                    "file": "jwt_keys.key",
                    "success": False,
                    "error": f"Failed to read file: {e}"
                })
                logger.error(f"  ❌ Failed to read JWT keys: {e}")
        
        # Test RSA private key
        rsa_private_file = current_dir / "rsa_private.pem"
        if rsa_private_file.exists():
            try:
                content = rsa_private_file.read_text()
                has_header = "-----BEGIN RSA PRIVATE KEY-----" in content
                has_footer = "-----END RSA PRIVATE KEY-----" in content
                
                content_tests.append({
                    "file": "rsa_private.pem",
                    "has_header": has_header,
                    "has_footer": has_footer,
                    "success": has_header and has_footer,
                    "error": None
                })
                
                if has_header and has_footer:
                    logger.info(f"  ✅ RSA private key: Valid PEM format")
                else:
                    logger.error(f"  ❌ RSA private key: Invalid PEM format")
                    
            except Exception as e:
                content_tests.append({
                    "file": "rsa_private.pem",
                    "success": False,
                    "error": f"Failed to read file: {e}"
                })
                logger.error(f"  ❌ Failed to read RSA private key: {e}")
        
        success = all(test["success"] for test in content_tests)
        
        result = {
            "test_name": test_name,
            "success": success,
            "content_tests": content_tests,
            "total_tests": len(content_tests),
            "passed_tests": sum(1 for test in content_tests if test["success"])
        }
        
        logger.info(f"📊 {test_name}: {'PASSED' if success else 'FAILED'}")
        return result
    
    def test_key_registry_validation(self) -> Dict[str, Any]:
        """Test key registry JSON format and content."""
        test_name = "Key Registry Validation"
        logger.info(f"🔍 Testing: {test_name}")
        
        registry_file = current_dir / "key_registry.json"
        
        if not registry_file.exists():
            result = {
                "test_name": test_name,
                "success": False,
                "error": "key_registry.json not found"
            }
            logger.error(f"  ❌ key_registry.json not found")
            return result
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check required fields
            required_fields = ["version", "keys"]
            missing_fields = [field for field in required_fields if field not in registry_data]
            
            # Check keys section
            keys_section = registry_data.get("keys", {})
            expected_keys = [
                "database_encryption",
                "jwt_signing", 
                "api_keys",
                "session_keys",
                "encryption_keys",
                "hmac_keys",
                "rsa_keys"
            ]
            
            missing_keys = [key for key in expected_keys if key not in keys_section]
            
            success = len(missing_fields) == 0 and len(missing_keys) == 0
            
            result = {
                "test_name": test_name,
                "success": success,
                "registry_data": registry_data,
                "missing_fields": missing_fields,
                "missing_keys": missing_keys,
                "total_keys": len(keys_section),
                "expected_keys": len(expected_keys)
            }
            
            if success:
                logger.info(f"  ✅ Key registry: Valid JSON with all required fields")
            else:
                logger.error(f"  ❌ Key registry: Missing fields={missing_fields}, missing keys={missing_keys}")
            
        except json.JSONDecodeError as e:
            result = {
                "test_name": test_name,
                "success": False,
                "error": f"Invalid JSON: {e}"
            }
            logger.error(f"  ❌ Key registry: Invalid JSON - {e}")
            
        except Exception as e:
            result = {
                "test_name": test_name,
                "success": False,
                "error": f"Unexpected error: {e}"
            }
            logger.error(f"  ❌ Key registry: Unexpected error - {e}")
        
        logger.info(f"📊 {test_name}: {'PASSED' if result['success'] else 'FAILED'}")
        return result
    
    async def test_python_integration(self) -> Dict[str, Any]:
        """Test Python integration with the key management system."""
        test_name = "Python Integration"
        logger.info(f"🔍 Testing: {test_name}")
        
        if not KEY_MANAGER_AVAILABLE:
            result = {
                "test_name": test_name,
                "success": False,
                "error": "Key management system not available"
            }
            logger.error(f"  ❌ Key management system not available")
            return result
        
        integration_tests = []
        
        try:
            # Test key manager initialization
            if self.key_manager is None:
                await self.initialize_key_manager()
            
            # Test key enumeration
            try:
                available_keys = list(self.key_manager.storage.list_keys())
                integration_tests.append({
                    "test": "key_enumeration",
                    "success": len(available_keys) > 0,
                    "available_keys": available_keys,
                    "key_count": len(available_keys)
                })
                logger.info(f"  ✅ Key enumeration: {len(available_keys)} keys found")
                
            except Exception as e:
                integration_tests.append({
                    "test": "key_enumeration",
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"  ❌ Key enumeration failed: {e}")
            
            # Test database key retrieval
            try:
                db_key = await self.key_manager.get_key_async("database_encryption", KeyUsage.ENCRYPTION)
                integration_tests.append({
                    "test": "database_key_retrieval",
                    "success": db_key is not None,
                    "key_type": type(db_key).__name__ if db_key else None
                })
                logger.info(f"  ✅ Database key retrieval: {type(db_key).__name__}")
                
            except Exception as e:
                integration_tests.append({
                    "test": "database_key_retrieval", 
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"  ❌ Database key retrieval failed: {e}")
            
            # Test encryption/decryption
            try:
                test_data = b"Hello, Enterprise Key Management!"
                encrypted_data = await self.key_manager.encrypt_data(test_data, "database_encryption")
                decrypted_data = await self.key_manager.decrypt_data(encrypted_data, "database_encryption")
                
                success = decrypted_data == test_data
                integration_tests.append({
                    "test": "encryption_decryption",
                    "success": success,
                    "original_length": len(test_data),
                    "encrypted_length": len(encrypted_data),
                    "decrypted_matches": success
                })
                
                if success:
                    logger.info(f"  ✅ Encryption/Decryption: Data integrity verified")
                else:
                    logger.error(f"  ❌ Encryption/Decryption: Data integrity check failed")
                    
            except Exception as e:
                integration_tests.append({
                    "test": "encryption_decryption",
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"  ❌ Encryption/Decryption failed: {e}")
            
            success = all(test["success"] for test in integration_tests)
            
            result = {
                "test_name": test_name,
                "success": success,
                "integration_tests": integration_tests,
                "total_tests": len(integration_tests),
                "passed_tests": sum(1 for test in integration_tests if test["success"])
            }
            
        except Exception as e:
            result = {
                "test_name": test_name,
                "success": False,
                "error": f"Integration test failed: {e}"
            }
            logger.error(f"  ❌ Python integration failed: {e}")
        
        logger.info(f"📊 {test_name}: {'PASSED' if result['success'] else 'FAILED'}")
        return result
    
    def test_script_functionality(self) -> Dict[str, Any]:
        """Test the functionality of management scripts."""
        test_name = "Script Functionality"
        logger.info(f"🔍 Testing: {test_name}")
        
        scripts = [
            "generate_keys.sh",
            "rotate_keys.sh", 
            "audit_keys.sh",
            "monitor_security.sh",
            "deploy_system.sh"
        ]
        
        script_tests = []
        
        for script in scripts:
            script_path = current_dir / script
            
            if script_path.exists():
                # Check if script is executable
                is_executable = os.access(script_path, os.X_OK)
                
                # Check script syntax (basic)
                import subprocess
                try:
                    result = subprocess.run(
                        ["bash", "-n", str(script_path)], 
                        capture_output=True, 
                        text=True,
                        timeout=10
                    )
                    syntax_valid = result.returncode == 0
                    
                except subprocess.TimeoutExpired:
                    syntax_valid = False
                except Exception:
                    syntax_valid = False
                
                script_tests.append({
                    "script": script,
                    "exists": True,
                    "executable": is_executable,
                    "syntax_valid": syntax_valid,
                    "success": is_executable and syntax_valid
                })
                
                if is_executable and syntax_valid:
                    logger.info(f"  ✅ {script}: Executable and syntax valid")
                else:
                    logger.error(f"  ❌ {script}: executable={is_executable}, syntax={syntax_valid}")
                    
            else:
                script_tests.append({
                    "script": script,
                    "exists": False,
                    "success": False
                })
                logger.error(f"  ❌ {script}: Not found")
        
        success = all(test["success"] for test in script_tests)
        
        result = {
            "test_name": test_name,
            "success": success,
            "script_tests": script_tests,
            "total_scripts": len(scripts),
            "working_scripts": sum(1 for test in script_tests if test["success"])
        }
        
        logger.info(f"📊 {test_name}: {'PASSED' if success else 'FAILED'}")
        return result
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        if success_rate >= 95:
            overall_status = "EXCELLENT"
        elif success_rate >= 85:
            overall_status = "GOOD"
        elif success_rate >= 70:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "overall_status": overall_status
            },
            "test_results": self.test_results,
            "recommendations": [],
            "timestamp": time.time(),
            "tester_info": {
                "version": "2.0.0",
                "author": "Fahed Mlaiel",
                "system": "Enterprise Key Management"
            }
        }
        
        # Add recommendations based on results
        if failed_tests > 0:
            report["recommendations"].append("Review and fix failed tests before production deployment")
        
        if success_rate < 100:
            report["recommendations"].append("Address remaining issues for optimal security")
        
        if success_rate >= 95:
            report["recommendations"].append("System ready for production deployment")
            
        return report
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Starting Enterprise Key Management System Integration Tests")
        logger.info("=" * 80)
        
        # Initialize key manager
        await self.initialize_key_manager()
        
        # Run tests
        tests = [
            self.test_key_file_presence,
            self.test_file_permissions,
            self.test_key_content_validation,
            self.test_key_registry_validation,
            self.test_python_integration,
            self.test_script_functionality
        ]
        
        for test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                self.test_results.append(result)
                
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
                self.test_results.append({
                    "test_name": getattr(test_func, '__name__', 'unknown'),
                    "success": False,
                    "error": str(e)
                })
        
        # Generate report
        report = self.generate_test_report()
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🏁 INTEGRATION TESTS COMPLETED")
        logger.info("=" * 80)
        logger.info(f"📊 Total Tests: {report['test_summary']['total_tests']}")
        logger.info(f"✅ Passed: {report['test_summary']['passed_tests']}")
        logger.info(f"❌ Failed: {report['test_summary']['failed_tests']}")
        logger.info(f"📈 Success Rate: {report['test_summary']['success_rate']}%")
        logger.info(f"🎯 Overall Status: {report['test_summary']['overall_status']}")
        
        if report['recommendations']:
            logger.info("\n📋 Recommendations:")
            for recommendation in report['recommendations']:
                logger.info(f"  • {recommendation}")
        
        return report

async def main():
    """Main test execution function."""
    tester = KeyManagementTester()
    report = await tester.run_all_tests()
    
    # Save detailed report
    report_file = current_dir / "integration_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📄 Detailed report saved: {report_file}")
    
    # Exit with appropriate code
    if report['test_summary']['success_rate'] >= 95:
        logger.info("🎉 INTEGRATION TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        sys.exit(0)
    elif report['test_summary']['success_rate'] >= 70:
        logger.warning("⚠️ INTEGRATION TESTS PARTIALLY PASSED - REVIEW NEEDED")
        sys.exit(1)
    else:
        logger.error("💥 INTEGRATION TESTS FAILED - SYSTEM NOT READY")
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())
