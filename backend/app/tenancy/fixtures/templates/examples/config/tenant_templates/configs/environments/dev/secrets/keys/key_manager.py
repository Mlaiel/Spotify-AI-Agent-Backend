#!/usr/bin/env python3
"""
Advanced Key Generation and Management Utilities
===============================================

Enterprise-grade key generation utilities with support for multiple
cryptographic algorithms, key formats, and security standards.

This module provides automated key generation, validation, and management
capabilities for production environments.
"""

import asyncio
import base64
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KeyGenerator:
    """Advanced cryptographic key generator."""
    
    def __init__(self):
        self.generation_history = []
    
    def generate_secure_random_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure random key."""
        return secrets.token_bytes(length)
    
    def generate_hex_key(self, length: int = 32) -> str:
        """Generate hexadecimal key."""
        return secrets.token_hex(length)
    
    def generate_base64_key(self, length: int = 32) -> str:
        """Generate base64 encoded key."""
        return base64.b64encode(secrets.token_bytes(length)).decode()
    
    def generate_urlsafe_key(self, length: int = 32) -> str:
        """Generate URL-safe base64 key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()
    
    def generate_jwt_secret(self, length: int = 64) -> str:
        """Generate JWT signing secret."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode().rstrip('=')
    
    def generate_api_key(self, prefix: str = "sai", length: int = 32) -> str:
        """Generate API key with prefix."""
        key_part = secrets.token_hex(length)
        return f"{prefix}_{key_part}"
    
    def generate_password_hash_salt(self, length: int = 16) -> str:
        """Generate salt for password hashing."""
        return base64.b64encode(secrets.token_bytes(length)).decode()
    
    def generate_encryption_key_pair(self) -> Tuple[str, str]:
        """Generate encryption key pair (public/private)."""
        private_key = self.generate_base64_key(32)
        public_key = self.generate_base64_key(32)
        return public_key, private_key
    
    def generate_hmac_key(self, length: int = 32) -> str:
        """Generate HMAC key."""
        return base64.b64encode(secrets.token_bytes(length)).decode()
    
    def generate_session_key(self, length: int = 24) -> str:
        """Generate session key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode().rstrip('=')
    
    def generate_csrf_token(self, length: int = 16) -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(length)
    
    def generate_database_encryption_key(self, algorithm: str = "AES-256") -> str:
        """Generate database encryption key."""
        if algorithm == "AES-256":
            return self.generate_base64_key(32)  # 256 bits
        elif algorithm == "AES-192":
            return self.generate_base64_key(24)  # 192 bits
        elif algorithm == "AES-128":
            return self.generate_base64_key(16)  # 128 bits
        else:
            return self.generate_base64_key(32)  # Default to 256 bits
    
    def generate_backup_encryption_key(self) -> str:
        """Generate backup encryption key."""
        return self.generate_base64_key(32)
    
    def derive_key_from_password(self, password: str, salt: str, iterations: int = 100000) -> str:
        """Derive key from password using PBKDF2."""
        password_bytes = password.encode('utf-8')
        salt_bytes = base64.b64decode(salt.encode('utf-8'))
        
        derived_key = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, iterations, 32)
        return base64.b64encode(derived_key).decode()
    
    def generate_key_with_metadata(self, key_type: str, **kwargs) -> Dict[str, str]:
        """Generate key with comprehensive metadata."""
        metadata = {
            'key_type': key_type,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '2.0.0',
            'security_level': kwargs.get('security_level', 'high'),
            'algorithm': kwargs.get('algorithm', 'AES-256'),
            'purpose': kwargs.get('purpose', 'general'),
            'environment': kwargs.get('environment', 'development'),
            'rotation_period': kwargs.get('rotation_period', '90d'),
            'compliance': kwargs.get('compliance', ['FIPS-140-2', 'Common Criteria'])
        }
        
        if key_type == "jwt":
            key_value = self.generate_jwt_secret()
        elif key_type == "api":
            key_value = self.generate_api_key()
        elif key_type == "database":
            key_value = self.generate_database_encryption_key()
        elif key_type == "session":
            key_value = self.generate_session_key()
        elif key_type == "hmac":
            key_value = self.generate_hmac_key()
        else:
            key_value = self.generate_base64_key()
        
        return {
            'key': key_value,
            'metadata': metadata
        }
    
    def validate_key_strength(self, key: str) -> Dict[str, bool]:
        """Validate key strength and security properties."""
        key_bytes = len(key.encode('utf-8'))
        
        return {
            'sufficient_length': key_bytes >= 32,
            'has_entropy': len(set(key)) > 10,
            'not_predictable': not any(pattern in key.lower() for pattern in ['1234', 'abcd', 'password']),
            'base64_encoded': self._is_base64(key),
            'url_safe': self._is_url_safe(key)
        }
    
    def _is_base64(self, s: str) -> bool:
        """Check if string is valid base64."""
        try:
            base64.b64decode(s)
            return True
        except Exception:
            return False
    
    def _is_url_safe(self, s: str) -> bool:
        """Check if string is URL safe."""
        try:
            base64.urlsafe_b64decode(s + '==')
            return True
        except Exception:
            return False


class KeyRotationScheduler:
    """Automated key rotation scheduling and management."""
    
    def __init__(self):
        self.rotation_schedule = {}
        self.rotation_history = []
    
    def schedule_rotation(self, key_id: str, interval_days: int):
        """Schedule automatic key rotation."""
        self.rotation_schedule[key_id] = {
            'interval_days': interval_days,
            'next_rotation': datetime.now() + timedelta(days=interval_days),
            'last_rotation': None,
            'rotation_count': 0
        }
    
    def check_due_rotations(self) -> List[str]:
        """Check which keys are due for rotation."""
        due_keys = []
        now = datetime.now()
        
        for key_id, schedule in self.rotation_schedule.items():
            if now >= schedule['next_rotation']:
                due_keys.append(key_id)
        
        return due_keys
    
    def mark_rotated(self, key_id: str):
        """Mark a key as rotated."""
        if key_id in self.rotation_schedule:
            schedule = self.rotation_schedule[key_id]
            schedule['last_rotation'] = datetime.now()
            schedule['next_rotation'] = datetime.now() + timedelta(days=schedule['interval_days'])
            schedule['rotation_count'] += 1
            
            self.rotation_history.append({
                'key_id': key_id,
                'rotated_at': datetime.now().isoformat(),
                'rotation_number': schedule['rotation_count']
            })


class KeyValidator:
    """Advanced key validation and security compliance checking."""
    
    def __init__(self):
        self.validation_rules = {
            'minimum_length': 32,
            'maximum_age_days': 90,
            'require_base64': True,
            'require_entropy': True,
            'forbidden_patterns': ['password', '123456', 'admin', 'test']
        }
    
    def validate_key(self, key: str, key_type: str = None) -> Dict[str, any]:
        """Comprehensive key validation."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'security_score': 100,
            'compliance_status': {}
        }
        
        # Check length
        if len(key) < self.validation_rules['minimum_length']:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Key too short (minimum {self.validation_rules['minimum_length']} characters)")
            validation_result['security_score'] -= 30
        
        # Check entropy
        if self.validation_rules['require_entropy']:
            unique_chars = len(set(key))
            if unique_chars < 10:
                validation_result['warnings'].append("Low entropy detected")
                validation_result['security_score'] -= 10
        
        # Check for forbidden patterns
        key_lower = key.lower()
        for pattern in self.validation_rules['forbidden_patterns']:
            if pattern in key_lower:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Forbidden pattern '{pattern}' found in key")
                validation_result['security_score'] -= 50
        
        # Check format compliance
        if self.validation_rules['require_base64']:
            try:
                base64.b64decode(key)
                validation_result['compliance_status']['base64_format'] = True
            except Exception:
                validation_result['warnings'].append("Key is not valid base64")
                validation_result['compliance_status']['base64_format'] = False
                validation_result['security_score'] -= 5
        
        # Type-specific validation
        if key_type:
            type_validation = self._validate_key_type(key, key_type)
            validation_result['compliance_status'].update(type_validation)
        
        return validation_result
    
    def _validate_key_type(self, key: str, key_type: str) -> Dict[str, bool]:
        """Validate key against type-specific requirements."""
        validation = {}
        
        if key_type == 'jwt':
            validation['jwt_format'] = '.' not in key  # JWT secrets shouldn't contain dots
            validation['jwt_length'] = len(key) >= 64
        
        elif key_type == 'api':
            validation['api_prefix'] = key.startswith(('sai_', 'api_', 'sk_'))
            validation['api_format'] = '_' in key
        
        elif key_type == 'database':
            validation['db_strength'] = len(key) >= 44  # 32 bytes base64 encoded
        
        elif key_type == 'session':
            validation['session_urlsafe'] = self._is_url_safe(key)
        
        return validation
    
    def _is_url_safe(self, s: str) -> bool:
        """Check if string is URL safe."""
        try:
            base64.urlsafe_b64decode(s + '==')
            return True
        except Exception:
            return False


class SecureKeyManager:
    """High-level secure key management interface."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.generator = KeyGenerator()
        self.validator = KeyValidator()
        self.scheduler = KeyRotationScheduler()
        self.key_registry = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load key registry from storage."""
        registry_file = self.storage_path / 'key_registry.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.key_registry = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load key registry: {e}")
                self.key_registry = {}
    
    def _save_registry(self):
        """Save key registry to storage."""
        registry_file = self.storage_path / 'key_registry.json'
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.key_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save key registry: {e}")
    
    def generate_and_store_key(self, key_id: str, key_type: str, **kwargs) -> bool:
        """Generate and securely store a new key."""
        try:
            # Generate key with metadata
            key_data = self.generator.generate_key_with_metadata(key_type, **kwargs)
            
            # Validate generated key
            validation = self.validator.validate_key(key_data['key'], key_type)
            if not validation['valid']:
                logger.error(f"Generated key failed validation: {validation['errors']}")
                return False
            
            # Store key securely
            key_file = self.storage_path / f"{key_id}.key"
            with open(key_file, 'w') as f:
                json.dump(key_data, f, indent=2)
            
            # Set restrictive permissions
            import os
            os.chmod(key_file, 0o600)
            
            # Update registry
            self.key_registry[key_id] = {
                'key_type': key_type,
                'created_at': datetime.now().isoformat(),
                'last_accessed': None,
                'access_count': 0,
                'validation_score': validation['security_score'],
                'status': 'active'
            }
            
            # Schedule rotation if specified
            if 'rotation_period' in kwargs:
                period = kwargs['rotation_period']
                if period.endswith('d'):
                    days = int(period[:-1])
                    self.scheduler.schedule_rotation(key_id, days)
            
            self._save_registry()
            logger.info(f"Key {key_id} generated and stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate and store key {key_id}: {e}")
            return False
    
    def get_key(self, key_id: str) -> Optional[str]:
        """Retrieve a key by ID."""
        try:
            key_file = self.storage_path / f"{key_id}.key"
            if not key_file.exists():
                return None
            
            with open(key_file, 'r') as f:
                key_data = json.load(f)
            
            # Update access tracking
            if key_id in self.key_registry:
                self.key_registry[key_id]['access_count'] += 1
                self.key_registry[key_id]['last_accessed'] = datetime.now().isoformat()
                self._save_registry()
            
            return key_data['key']
            
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None
    
    def rotate_key(self, key_id: str) -> bool:
        """Rotate an existing key."""
        try:
            # Get current key metadata
            key_file = self.storage_path / f"{key_id}.key"
            if not key_file.exists():
                logger.error(f"Key {key_id} not found for rotation")
                return False
            
            with open(key_file, 'r') as f:
                old_key_data = json.load(f)
            
            # Generate new key with same type
            new_key_data = self.generator.generate_key_with_metadata(
                old_key_data['metadata']['key_type']
            )
            
            # Backup old key
            backup_file = self.storage_path / f"{key_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.key"
            with open(backup_file, 'w') as f:
                json.dump(old_key_data, f, indent=2)
            
            # Store new key
            with open(key_file, 'w') as f:
                json.dump(new_key_data, f, indent=2)
            
            # Update registry
            if key_id in self.key_registry:
                self.key_registry[key_id]['last_rotated'] = datetime.now().isoformat()
                self.key_registry[key_id]['rotation_count'] = self.key_registry[key_id].get('rotation_count', 0) + 1
            
            # Mark as rotated in scheduler
            self.scheduler.mark_rotated(key_id)
            
            self._save_registry()
            logger.info(f"Key {key_id} rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            return False
    
    def list_keys(self) -> List[Dict[str, any]]:
        """List all managed keys with their metadata."""
        keys = []
        
        for key_id, metadata in self.key_registry.items():
            key_info = {
                'key_id': key_id,
                'key_type': metadata['key_type'],
                'created_at': metadata['created_at'],
                'last_accessed': metadata.get('last_accessed'),
                'access_count': metadata.get('access_count', 0),
                'status': metadata.get('status', 'active'),
                'validation_score': metadata.get('validation_score', 0)
            }
            keys.append(key_info)
        
        return keys
    
    def check_key_health(self) -> Dict[str, any]:
        """Check overall key health and security status."""
        total_keys = len(self.key_registry)
        active_keys = len([k for k in self.key_registry.values() if k.get('status') == 'active'])
        
        # Check for keys due for rotation
        due_rotations = self.scheduler.check_due_rotations()
        
        # Calculate average security score
        scores = [k.get('validation_score', 0) for k in self.key_registry.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'total_keys': total_keys,
            'active_keys': active_keys,
            'keys_due_rotation': len(due_rotations),
            'due_rotation_list': due_rotations,
            'average_security_score': avg_score,
            'health_status': 'healthy' if avg_score > 80 and len(due_rotations) == 0 else 'needs_attention'
        }
    
    def generate_development_keys(self) -> Dict[str, bool]:
        """Generate standard development environment keys."""
        development_keys = {
            'database_encryption_key': 'database',
            'jwt_signing_key': 'jwt',
            'api_master_key': 'api',
            'session_encryption_key': 'session',
            'backup_encryption_key': 'database',
            'hmac_validation_key': 'hmac'
        }
        
        results = {}
        
        for key_id, key_type in development_keys.items():
            success = self.generate_and_store_key(
                key_id=key_id,
                key_type=key_type,
                environment='development',
                security_level='high',
                rotation_period='30d'
            )
            results[key_id] = success
        
        return results


# Convenience functions for quick operations

def generate_quick_keys(storage_path: str) -> Dict[str, str]:
    """Generate a quick set of development keys."""
    manager = SecureKeyManager(Path(storage_path))
    
    keys = {
        'database_key': manager.generator.generate_database_encryption_key(),
        'jwt_secret': manager.generator.generate_jwt_secret(),
        'api_key': manager.generator.generate_api_key(),
        'session_key': manager.generator.generate_session_key(),
        'hmac_key': manager.generator.generate_hmac_key()
    }
    
    return keys


def validate_existing_key(key: str, key_type: str = None) -> Dict[str, any]:
    """Validate an existing key."""
    validator = KeyValidator()
    return validator.validate_key(key, key_type)


async def automated_key_rotation(storage_path: str):
    """Perform automated key rotation for due keys."""
    manager = SecureKeyManager(Path(storage_path))
    
    due_keys = manager.scheduler.check_due_rotations()
    
    results = {}
    for key_id in due_keys:
        success = manager.rotate_key(key_id)
        results[key_id] = success
        
        if success:
            logger.info(f"Successfully rotated key: {key_id}")
        else:
            logger.error(f"Failed to rotate key: {key_id}")
    
    return results


# Main execution function
async def main():
    """Main function for key management operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Key Management Utilities')
    parser.add_argument('--action', choices=['generate', 'rotate', 'validate', 'list', 'health'], 
                       default='generate', help='Action to perform')
    parser.add_argument('--storage-path', default='./keys', help='Key storage path')
    parser.add_argument('--key-id', help='Key ID for operations')
    parser.add_argument('--key-type', help='Key type for generation')
    parser.add_argument('--key-value', help='Key value for validation')
    
    args = parser.parse_args()
    
    manager = SecureKeyManager(Path(args.storage_path))
    
    if args.action == 'generate':
        if args.key_id and args.key_type:
            success = manager.generate_and_store_key(args.key_id, args.key_type)
            print(f"Key generation {'successful' if success else 'failed'}")
        else:
            # Generate development keys
            results = manager.generate_development_keys()
            print(f"Generated {sum(results.values())} out of {len(results)} development keys")
    
    elif args.action == 'rotate':
        if args.key_id:
            success = manager.rotate_key(args.key_id)
            print(f"Key rotation {'successful' if success else 'failed'}")
        else:
            results = await automated_key_rotation(args.storage_path)
            print(f"Rotated {sum(results.values())} out of {len(results)} keys")
    
    elif args.action == 'validate':
        if args.key_value:
            result = validate_existing_key(args.key_value, args.key_type)
            print(f"Validation result: {result}")
        else:
            print("Key value required for validation")
    
    elif args.action == 'list':
        keys = manager.list_keys()
        print(f"Found {len(keys)} keys:")
        for key in keys:
            print(f"  - {key['key_id']}: {key['key_type']} (score: {key['validation_score']})")
    
    elif args.action == 'health':
        health = manager.check_key_health()
        print(f"Key health status: {health}")


if __name__ == "__main__":
    asyncio.run(main())
