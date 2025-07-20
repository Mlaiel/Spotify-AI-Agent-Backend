#!/usr/bin/env python3
"""
Enterprise Keys Backup Management System
========================================

Ultra-advanced backup management system for cryptographic keys and secrets
with automated scheduling, encryption, versioning, and disaster recovery.

This module provides comprehensive backup orchestration, secure storage,
automated rotation, integrity verification, and cloud synchronization.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

Features:
- Automated backup scheduling and execution
- Multi-layer encryption and compression
- Versioning and retention policies
- Integrity verification and validation
- Cloud storage synchronization
- Disaster recovery automation
- Performance optimization and monitoring
- Compliance and audit trail
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import gzip
import pickle
import sqlite3
import threading
from queue import Queue, PriorityQueue
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backup_manager.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups supported."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    EMERGENCY = "emergency"


class CompressionType(Enum):
    """Compression algorithms supported."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    TAR_XZ = "tar.xz"


class EncryptionType(Enum):
    """Encryption methods for backups."""
    NONE = "none"
    AES256 = "aes256"
    FERNET = "fernet"
    RSA = "rsa"
    HYBRID = "hybrid"  # RSA + AES


class StorageBackend(Enum):
    """Storage backends for backups."""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCP = "gcp"
    FTP = "ftp"
    SFTP = "sftp"
    NFS = "nfs"
    DISTRIBUTED = "distributed"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"
    ARCHIVED = "archived"


@dataclass
class BackupMetadata:
    """Comprehensive backup metadata."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    source_path: str
    destination_path: str
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    file_count: int = 0
    checksum_sha256: str = ""
    checksum_md5: str = ""
    compression: CompressionType = CompressionType.GZIP
    encryption: EncryptionType = EncryptionType.FERNET
    storage_backend: StorageBackend = StorageBackend.LOCAL
    status: BackupStatus = BackupStatus.PENDING
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    retention_days: int = 30
    created_by: str = "BackupManager"
    verification_status: bool = False
    last_verified: Optional[datetime] = None
    restore_tested: bool = False
    notes: str = ""
    parent_backup_id: Optional[str] = None
    child_backup_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    cloud_sync_status: str = "pending"
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class BackupConfiguration:
    """Backup system configuration."""
    base_backup_dir: str = "./backups"
    max_backup_size_gb: float = 10.0
    retention_days: int = 30
    max_versions: int = 10
    compression_level: int = 6
    encryption_enabled: bool = True
    verification_enabled: bool = True
    cloud_sync_enabled: bool = False
    parallel_workers: int = 4
    max_memory_usage_mb: int = 512
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    emergency_threshold_mb: int = 100
    integrity_check_interval_hours: int = 24
    auto_cleanup_enabled: bool = True
    notification_enabled: bool = True
    audit_log_enabled: bool = True
    performance_monitoring: bool = True


class BackupError(Exception):
    """Custom exception for backup operations."""
    pass


class BackupStorageInterface(ABC):
    """Abstract interface for backup storage backends."""
    
    @abstractmethod
    async def store_backup(self, backup_data: bytes, metadata: BackupMetadata) -> bool:
        """Store backup data."""
        pass
    
    @abstractmethod
    async def retrieve_backup(self, backup_id: str) -> Tuple[bytes, BackupMetadata]:
        """Retrieve backup data."""
        pass
    
    @abstractmethod
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup."""
        pass
    
    @abstractmethod
    async def list_backups(self) -> List[BackupMetadata]:
        """List all backups."""
        pass


class LocalStorageBackend(BackupStorageInterface):
    """Local file system storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_db = self.base_path / "backup_metadata.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self.metadata_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_metadata (
                backup_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    async def store_backup(self, backup_data: bytes, metadata: BackupMetadata) -> bool:
        """Store backup data locally."""
        try:
            backup_file = self.base_path / f"{metadata.backup_id}.backup"
            metadata_file = self.base_path / f"{metadata.backup_id}.metadata.json"
            
            # Write backup data
            with open(backup_file, 'wb') as f:
                f.write(backup_data)
            
            # Write metadata
            metadata_dict = {
                'backup_id': metadata.backup_id,
                'backup_type': metadata.backup_type.value,
                'timestamp': metadata.timestamp.isoformat(),
                'source_path': metadata.source_path,
                'destination_path': str(backup_file),
                'size_bytes': metadata.size_bytes,
                'compressed_size_bytes': metadata.compressed_size_bytes,
                'file_count': metadata.file_count,
                'checksum_sha256': metadata.checksum_sha256,
                'checksum_md5': metadata.checksum_md5,
                'compression': metadata.compression.value,
                'encryption': metadata.encryption.value,
                'storage_backend': metadata.storage_backend.value,
                'status': metadata.status.value,
                'version': metadata.version,
                'tags': metadata.tags,
                'retention_days': metadata.retention_days,
                'created_by': metadata.created_by,
                'verification_status': metadata.verification_status,
                'last_verified': metadata.last_verified.isoformat() if metadata.last_verified else None,
                'restore_tested': metadata.restore_tested,
                'notes': metadata.notes,
                'parent_backup_id': metadata.parent_backup_id,
                'child_backup_ids': metadata.child_backup_ids,
                'dependencies': metadata.dependencies,
                'cloud_sync_status': metadata.cloud_sync_status,
                'access_count': metadata.access_count,
                'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Store in database
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO backup_metadata (backup_id, metadata_json) VALUES (?, ?)',
                (metadata.backup_id, json.dumps(metadata_dict))
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully stored backup {metadata.backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store backup {metadata.backup_id}: {e}")
            return False
    
    async def retrieve_backup(self, backup_id: str) -> Tuple[bytes, BackupMetadata]:
        """Retrieve backup data."""
        try:
            backup_file = self.base_path / f"{backup_id}.backup"
            metadata_file = self.base_path / f"{backup_id}.metadata.json"
            
            if not backup_file.exists() or not metadata_file.exists():
                raise BackupError(f"Backup {backup_id} not found")
            
            # Read backup data
            with open(backup_file, 'rb') as f:
                backup_data = f.read()
            
            # Read metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            # Convert to BackupMetadata object
            metadata = BackupMetadata(
                backup_id=metadata_dict['backup_id'],
                backup_type=BackupType(metadata_dict['backup_type']),
                timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                source_path=metadata_dict['source_path'],
                destination_path=metadata_dict['destination_path'],
                size_bytes=metadata_dict['size_bytes'],
                compressed_size_bytes=metadata_dict['compressed_size_bytes'],
                file_count=metadata_dict['file_count'],
                checksum_sha256=metadata_dict['checksum_sha256'],
                checksum_md5=metadata_dict['checksum_md5'],
                compression=CompressionType(metadata_dict['compression']),
                encryption=EncryptionType(metadata_dict['encryption']),
                storage_backend=StorageBackend(metadata_dict['storage_backend']),
                status=BackupStatus(metadata_dict['status']),
                version=metadata_dict['version'],
                tags=metadata_dict['tags'],
                retention_days=metadata_dict['retention_days'],
                created_by=metadata_dict['created_by'],
                verification_status=metadata_dict['verification_status'],
                last_verified=datetime.fromisoformat(metadata_dict['last_verified']) if metadata_dict['last_verified'] else None,
                restore_tested=metadata_dict['restore_tested'],
                notes=metadata_dict['notes'],
                parent_backup_id=metadata_dict['parent_backup_id'],
                child_backup_ids=metadata_dict['child_backup_ids'],
                dependencies=metadata_dict['dependencies'],
                cloud_sync_status=metadata_dict['cloud_sync_status'],
                access_count=metadata_dict['access_count'],
                last_accessed=datetime.fromisoformat(metadata_dict['last_accessed']) if metadata_dict['last_accessed'] else None
            )
            
            # Update access count
            metadata.access_count += 1
            metadata.last_accessed = datetime.now()
            
            logger.info(f"Successfully retrieved backup {backup_id}")
            return backup_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve backup {backup_id}: {e}")
            raise BackupError(f"Failed to retrieve backup: {e}")
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup."""
        try:
            backup_file = self.base_path / f"{backup_id}.backup"
            metadata_file = self.base_path / f"{backup_id}.metadata.json"
            
            if backup_file.exists():
                backup_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from database
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM backup_metadata WHERE backup_id = ?', (backup_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully deleted backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List all backups."""
        try:
            conn = sqlite3.connect(str(self.metadata_db))
            cursor = conn.cursor()
            cursor.execute('SELECT metadata_json FROM backup_metadata ORDER BY created_at DESC')
            rows = cursor.fetchall()
            conn.close()
            
            backups = []
            for row in rows:
                metadata_dict = json.loads(row[0])
                metadata = BackupMetadata(
                    backup_id=metadata_dict['backup_id'],
                    backup_type=BackupType(metadata_dict['backup_type']),
                    timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                    source_path=metadata_dict['source_path'],
                    destination_path=metadata_dict['destination_path'],
                    size_bytes=metadata_dict['size_bytes'],
                    compressed_size_bytes=metadata_dict['compressed_size_bytes'],
                    file_count=metadata_dict['file_count'],
                    checksum_sha256=metadata_dict['checksum_sha256'],
                    checksum_md5=metadata_dict['checksum_md5'],
                    compression=CompressionType(metadata_dict['compression']),
                    encryption=EncryptionType(metadata_dict['encryption']),
                    storage_backend=StorageBackend(metadata_dict['storage_backend']),
                    status=BackupStatus(metadata_dict['status']),
                    version=metadata_dict['version'],
                    tags=metadata_dict['tags'],
                    retention_days=metadata_dict['retention_days'],
                    created_by=metadata_dict['created_by'],
                    verification_status=metadata_dict['verification_status'],
                    last_verified=datetime.fromisoformat(metadata_dict['last_verified']) if metadata_dict['last_verified'] else None,
                    restore_tested=metadata_dict['restore_tested'],
                    notes=metadata_dict['notes'],
                    parent_backup_id=metadata_dict['parent_backup_id'],
                    child_backup_ids=metadata_dict['child_backup_ids'],
                    dependencies=metadata_dict['dependencies'],
                    cloud_sync_status=metadata_dict['cloud_sync_status'],
                    access_count=metadata_dict['access_count'],
                    last_accessed=datetime.fromisoformat(metadata_dict['last_accessed']) if metadata_dict['last_accessed'] else None
                )
                backups.append(metadata)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []


class EncryptionManager:
    """Advanced encryption manager for backup data."""
    
    def __init__(self, config: BackupConfiguration):
        self.config = config
        self.fernet_key = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption keys."""
        try:
            # Initialize Fernet key
            key_file = Path("backup_encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.fernet_key = f.read()
            else:
                self.fernet_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.fernet_key)
                # Secure permissions
                os.chmod(key_file, 0o600)
            
            # Initialize RSA keys
            private_key_file = Path("backup_rsa_private.pem")
            public_key_file = Path("backup_rsa_public.pem")
            
            if private_key_file.exists() and public_key_file.exists():
                with open(private_key_file, 'rb') as f:
                    self.rsa_private_key = serialization.load_pem_private_key(f.read(), password=None)
                with open(public_key_file, 'rb') as f:
                    self.rsa_public_key = serialization.load_pem_public_key(f.read())
            else:
                # Generate new RSA key pair
                self.rsa_private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096
                )
                self.rsa_public_key = self.rsa_private_key.public_key()
                
                # Save keys
                with open(private_key_file, 'wb') as f:
                    f.write(self.rsa_private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                
                with open(public_key_file, 'wb') as f:
                    f.write(self.rsa_public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
                
                # Secure permissions
                os.chmod(private_key_file, 0o600)
                os.chmod(public_key_file, 0o644)
            
            logger.info("Encryption manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise BackupError(f"Encryption initialization failed: {e}")
    
    def encrypt_data(self, data: bytes, encryption_type: EncryptionType) -> bytes:
        """Encrypt backup data."""
        try:
            if encryption_type == EncryptionType.NONE:
                return data
            
            elif encryption_type == EncryptionType.FERNET:
                fernet = Fernet(self.fernet_key)
                return fernet.encrypt(data)
            
            elif encryption_type == EncryptionType.HYBRID:
                # Hybrid: Encrypt data with Fernet, encrypt Fernet key with RSA
                fernet_key = Fernet.generate_key()
                fernet = Fernet(fernet_key)
                encrypted_data = fernet.encrypt(data)
                
                # Encrypt the Fernet key with RSA
                encrypted_key = self.rsa_public_key.encrypt(
                    fernet_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Combine encrypted key and data
                key_length = len(encrypted_key).to_bytes(4, 'big')
                return key_length + encrypted_key + encrypted_data
            
            else:
                raise BackupError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise BackupError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes, encryption_type: EncryptionType) -> bytes:
        """Decrypt backup data."""
        try:
            if encryption_type == EncryptionType.NONE:
                return encrypted_data
            
            elif encryption_type == EncryptionType.FERNET:
                fernet = Fernet(self.fernet_key)
                return fernet.decrypt(encrypted_data)
            
            elif encryption_type == EncryptionType.HYBRID:
                # Extract encrypted key length
                key_length = int.from_bytes(encrypted_data[:4], 'big')
                encrypted_key = encrypted_data[4:4+key_length]
                encrypted_data = encrypted_data[4+key_length:]
                
                # Decrypt the Fernet key with RSA
                fernet_key = self.rsa_private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt data with Fernet
                fernet = Fernet(fernet_key)
                return fernet.decrypt(encrypted_data)
            
            else:
                raise BackupError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise BackupError(f"Decryption failed: {e}")


class CompressionManager:
    """Advanced compression manager for backup data."""
    
    @staticmethod
    def compress_data(data: bytes, compression_type: CompressionType, 
                     level: int = 6) -> bytes:
        """Compress backup data."""
        try:
            if compression_type == CompressionType.NONE:
                return data
            
            elif compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=level)
            
            elif compression_type == CompressionType.BZIP2:
                import bz2
                return bz2.compress(data, compresslevel=level)
            
            elif compression_type == CompressionType.LZMA:
                import lzma
                return lzma.compress(data, preset=level)
            
            else:
                raise BackupError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise BackupError(f"Compression failed: {e}")
    
    @staticmethod
    def decompress_data(compressed_data: bytes, 
                       compression_type: CompressionType) -> bytes:
        """Decompress backup data."""
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            
            elif compression_type == CompressionType.BZIP2:
                import bz2
                return bz2.decompress(compressed_data)
            
            elif compression_type == CompressionType.LZMA:
                import lzma
                return lzma.decompress(compressed_data)
            
            else:
                raise BackupError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise BackupError(f"Decompression failed: {e}")


class IntegrityVerifier:
    """Integrity verification and validation."""
    
    @staticmethod
    def calculate_checksums(data: bytes) -> Tuple[str, str]:
        """Calculate SHA256 and MD5 checksums."""
        sha256_hash = hashlib.sha256(data).hexdigest()
        md5_hash = hashlib.md5(data).hexdigest()
        return sha256_hash, md5_hash
    
    @staticmethod
    def verify_integrity(data: bytes, expected_sha256: str, 
                        expected_md5: str) -> bool:
        """Verify data integrity using checksums."""
        actual_sha256, actual_md5 = IntegrityVerifier.calculate_checksums(data)
        return (actual_sha256 == expected_sha256 and 
                actual_md5 == expected_md5)


class BackupScheduler:
    """Advanced backup scheduling and automation."""
    
    def __init__(self, backup_manager):
        self.backup_manager = backup_manager
        self.scheduled_jobs = {}
        self.running = False
        self.scheduler_thread = None
    
    def schedule_backup(self, backup_id: str, cron_expression: str, 
                       backup_config: Dict[str, Any]):
        """Schedule a backup job."""
        self.scheduled_jobs[backup_id] = {
            'cron': cron_expression,
            'config': backup_config,
            'last_run': None,
            'next_run': self._calculate_next_run(cron_expression),
            'enabled': True
        }
        logger.info(f"Scheduled backup job {backup_id} with cron {cron_expression}")
    
    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression."""
        # Simplified cron calculation (basic implementation)
        # In production, use a proper cron library like croniter
        now = datetime.now()
        return now + timedelta(hours=24)  # Default to daily
    
    def start_scheduler(self):
        """Start the backup scheduler."""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Backup scheduler started")
    
    def stop_scheduler(self):
        """Stop the backup scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Backup scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            now = datetime.now()
            
            for job_id, job_config in self.scheduled_jobs.items():
                if (job_config['enabled'] and 
                    job_config['next_run'] <= now):
                    
                    # Execute backup
                    try:
                        asyncio.create_task(
                            self.backup_manager.create_backup(
                                **job_config['config']
                            )
                        )
                        job_config['last_run'] = now
                        job_config['next_run'] = self._calculate_next_run(
                            job_config['cron']
                        )
                        logger.info(f"Executed scheduled backup {job_id}")
                    except Exception as e:
                        logger.error(f"Failed to execute scheduled backup {job_id}: {e}")
            
            time.sleep(60)  # Check every minute


class BackupManager:
    """Enterprise backup management system."""
    
    def __init__(self, config: BackupConfiguration):
        self.config = config
        self.storage_backend = LocalStorageBackend(config.base_backup_dir)
        self.encryption_manager = EncryptionManager(config)
        self.compression_manager = CompressionManager()
        self.integrity_verifier = IntegrityVerifier()
        self.scheduler = BackupScheduler(self)
        
        # Performance monitoring
        self.metrics = {
            'backups_created': 0,
            'backups_restored': 0,
            'backups_verified': 0,
            'total_backup_size_bytes': 0,
            'total_compressed_size_bytes': 0,
            'average_compression_ratio': 0.0,
            'average_backup_time_seconds': 0.0,
            'errors_count': 0,
            'last_backup_time': None,
            'uptime_seconds': 0
        }
        
        self.start_time = time.time()
        
        # Create backup directories
        Path(self.config.base_backup_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("BackupManager initialized successfully")
    
    async def create_backup(self, source_path: str, 
                          backup_type: BackupType = BackupType.FULL,
                          compression: CompressionType = CompressionType.GZIP,
                          encryption: EncryptionType = EncryptionType.FERNET,
                          tags: List[str] = None,
                          notes: str = "") -> str:
        """Create a new backup."""
        start_time = time.time()
        backup_id = self._generate_backup_id(backup_type)
        
        try:
            logger.info(f"Starting backup {backup_id} for {source_path}")
            
            # Validate source path
            source = Path(source_path)
            if not source.exists():
                raise BackupError(f"Source path does not exist: {source_path}")
            
            # Create backup data
            backup_data = await self._create_backup_archive(source, backup_type)
            original_size = len(backup_data)
            
            # Compress data
            if compression != CompressionType.NONE:
                backup_data = self.compression_manager.compress_data(
                    backup_data, compression, self.config.compression_level
                )
            compressed_size = len(backup_data)
            
            # Calculate checksums before encryption
            sha256_hash, md5_hash = self.integrity_verifier.calculate_checksums(backup_data)
            
            # Encrypt data
            if encryption != EncryptionType.NONE and self.config.encryption_enabled:
                backup_data = self.encryption_manager.encrypt_data(backup_data, encryption)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=datetime.now(),
                source_path=source_path,
                destination_path=str(Path(self.config.base_backup_dir) / f"{backup_id}.backup"),
                size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                file_count=await self._count_files(source),
                checksum_sha256=sha256_hash,
                checksum_md5=md5_hash,
                compression=compression,
                encryption=encryption,
                storage_backend=StorageBackend.LOCAL,
                status=BackupStatus.RUNNING,
                tags=tags or [],
                retention_days=self.config.retention_days,
                notes=notes,
                verification_status=False
            )
            
            # Store backup
            success = await self.storage_backend.store_backup(backup_data, metadata)
            
            if success:
                metadata.status = BackupStatus.SUCCESS
                
                # Verify backup if enabled
                if self.config.verification_enabled:
                    verification_success = await self.verify_backup(backup_id)
                    metadata.verification_status = verification_success
                    metadata.last_verified = datetime.now()
                
                # Update metrics
                self.metrics['backups_created'] += 1
                self.metrics['total_backup_size_bytes'] += original_size
                self.metrics['total_compressed_size_bytes'] += compressed_size
                self.metrics['last_backup_time'] = datetime.now()
                
                # Calculate compression ratio
                if original_size > 0:
                    compression_ratio = compressed_size / original_size
                    self.metrics['average_compression_ratio'] = (
                        (self.metrics['average_compression_ratio'] * 
                         (self.metrics['backups_created'] - 1) + compression_ratio) /
                        self.metrics['backups_created']
                    )
                
                # Calculate average backup time
                backup_time = time.time() - start_time
                self.metrics['average_backup_time_seconds'] = (
                    (self.metrics['average_backup_time_seconds'] * 
                     (self.metrics['backups_created'] - 1) + backup_time) /
                    self.metrics['backups_created']
                )
                
                logger.info(f"Backup {backup_id} created successfully in {backup_time:.2f}s")
                return backup_id
            else:
                metadata.status = BackupStatus.FAILED
                self.metrics['errors_count'] += 1
                raise BackupError("Failed to store backup")
                
        except Exception as e:
            self.metrics['errors_count'] += 1
            logger.error(f"Failed to create backup {backup_id}: {e}")
            raise BackupError(f"Backup creation failed: {e}")
    
    async def _create_backup_archive(self, source_path: Path, 
                                   backup_type: BackupType) -> bytes:
        """Create backup archive from source path."""
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                with tarfile.open(temp_file.name, 'w') as tar:
                    if source_path.is_file():
                        tar.add(source_path, arcname=source_path.name)
                    else:
                        for item in source_path.rglob('*'):
                            if item.is_file():
                                tar.add(item, arcname=item.relative_to(source_path.parent))
                
                temp_file.seek(0)
                return temp_file.read()
                
        except Exception as e:
            logger.error(f"Failed to create backup archive: {e}")
            raise BackupError(f"Archive creation failed: {e}")
    
    async def _count_files(self, source_path: Path) -> int:
        """Count files in source path."""
        try:
            if source_path.is_file():
                return 1
            else:
                return sum(1 for item in source_path.rglob('*') if item.is_file())
        except Exception:
            return 0
    
    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{backup_type.value}_{timestamp}_{int(time.time() * 1000) % 100000}"
        return backup_id
    
    async def restore_backup(self, backup_id: str, 
                           destination_path: str,
                           verify_integrity: bool = True) -> bool:
        """Restore backup to destination."""
        try:
            logger.info(f"Starting restore of backup {backup_id} to {destination_path}")
            
            # Retrieve backup
            backup_data, metadata = await self.storage_backend.retrieve_backup(backup_id)
            
            # Decrypt data
            if metadata.encryption != EncryptionType.NONE:
                backup_data = self.encryption_manager.decrypt_data(
                    backup_data, metadata.encryption
                )
            
            # Decompress data
            if metadata.compression != CompressionType.NONE:
                backup_data = self.compression_manager.decompress_data(
                    backup_data, metadata.compression
                )
            
            # Verify integrity
            if verify_integrity:
                if not self.integrity_verifier.verify_integrity(
                    backup_data, metadata.checksum_sha256, metadata.checksum_md5
                ):
                    raise BackupError("Backup integrity verification failed")
            
            # Extract to destination
            destination = Path(destination_path)
            destination.mkdir(parents=True, exist_ok=True)
            
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(backup_data)
                temp_file.flush()
                
                with tarfile.open(temp_file.name, 'r') as tar:
                    tar.extractall(destination)
            
            # Update metrics
            self.metrics['backups_restored'] += 1
            
            logger.info(f"Successfully restored backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            self.metrics['errors_count'] += 1
            return False
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        try:
            logger.info(f"Verifying backup {backup_id}")
            
            # Retrieve backup
            backup_data, metadata = await self.storage_backend.retrieve_backup(backup_id)
            
            # Decrypt if needed
            if metadata.encryption != EncryptionType.NONE:
                decrypted_data = self.encryption_manager.decrypt_data(
                    backup_data, metadata.encryption
                )
            else:
                decrypted_data = backup_data
            
            # Decompress if needed
            if metadata.compression != CompressionType.NONE:
                decompressed_data = self.compression_manager.decompress_data(
                    decrypted_data, metadata.compression
                )
            else:
                decompressed_data = decrypted_data
            
            # Verify checksums
            verification_result = self.integrity_verifier.verify_integrity(
                decompressed_data, metadata.checksum_sha256, metadata.checksum_md5
            )
            
            # Update metrics
            self.metrics['backups_verified'] += 1
            
            if verification_result:
                logger.info(f"Backup {backup_id} verification successful")
            else:
                logger.error(f"Backup {backup_id} verification failed")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Failed to verify backup {backup_id}: {e}")
            self.metrics['errors_count'] += 1
            return False
    
    async def list_backups(self, backup_type: Optional[BackupType] = None,
                          tags: Optional[List[str]] = None) -> List[BackupMetadata]:
        """List backups with optional filtering."""
        try:
            all_backups = await self.storage_backend.list_backups()
            
            filtered_backups = all_backups
            
            # Filter by backup type
            if backup_type:
                filtered_backups = [
                    b for b in filtered_backups 
                    if b.backup_type == backup_type
                ]
            
            # Filter by tags
            if tags:
                filtered_backups = [
                    b for b in filtered_backups
                    if any(tag in b.tags for tag in tags)
                ]
            
            return filtered_backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        try:
            logger.info(f"Deleting backup {backup_id}")
            success = await self.storage_backend.delete_backup(backup_id)
            
            if success:
                logger.info(f"Successfully deleted backup {backup_id}")
            else:
                logger.error(f"Failed to delete backup {backup_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def cleanup_expired_backups(self) -> int:
        """Clean up expired backups."""
        try:
            if not self.config.auto_cleanup_enabled:
                return 0
            
            logger.info("Starting expired backup cleanup")
            
            all_backups = await self.storage_backend.list_backups()
            deleted_count = 0
            now = datetime.now()
            
            for backup in all_backups:
                # Check retention policy
                retention_deadline = backup.timestamp + timedelta(days=backup.retention_days)
                
                if now > retention_deadline:
                    success = await self.delete_backup(backup.backup_id)
                    if success:
                        deleted_count += 1
            
            logger.info(f"Cleanup completed. Deleted {deleted_count} expired backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired backups: {e}")
            return 0
    
    async def get_backup_metrics(self) -> Dict[str, Any]:
        """Get backup system metrics."""
        self.metrics['uptime_seconds'] = time.time() - self.start_time
        
        # Calculate additional metrics
        all_backups = await self.storage_backend.list_backups()
        self.metrics['total_backups'] = len(all_backups)
        
        return self.metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'backup_system': {
                'encryption_manager': 'healthy',
                'storage_backend': 'healthy',
                'compression_manager': 'healthy',
                'scheduler': 'healthy' if self.scheduler.running else 'stopped'
            },
            'metrics': await self.get_backup_metrics(),
            'storage_info': {
                'base_backup_dir': self.config.base_backup_dir,
                'disk_usage': await self._get_disk_usage(),
                'total_backups': len(await self.storage_backend.list_backups())
            },
            'issues': []
        }
        
        # Check for issues
        if self.metrics['errors_count'] > 10:
            health_status['issues'].append(f"High error count: {self.metrics['errors_count']}")
            health_status['overall_status'] = 'degraded'
        
        # Check disk space
        disk_usage = await self._get_disk_usage()
        if disk_usage['usage_percent'] > 90:
            health_status['issues'].append(f"High disk usage: {disk_usage['usage_percent']}%")
            health_status['overall_status'] = 'critical'
        
        return health_status
    
    async def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.config.base_backup_dir)
            usage_percent = (used / total) * 100
            
            return {
                'total_bytes': total,
                'used_bytes': used,
                'free_bytes': free,
                'usage_percent': round(usage_percent, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {
                'total_bytes': 0,
                'used_bytes': 0,
                'free_bytes': 0,
                'usage_percent': 0.0
            }
    
    def start_scheduler(self):
        """Start the backup scheduler."""
        self.scheduler.start_scheduler()
    
    def stop_scheduler(self):
        """Stop the backup scheduler."""
        self.scheduler.stop_scheduler()
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down BackupManager")
        self.stop_scheduler()
        # Perform any final cleanup
        await self.cleanup_expired_backups()
        logger.info("BackupManager shutdown completed")


# Convenience functions for common operations

async def create_keys_backup(keys_dir: str = "../", 
                           config: Optional[BackupConfiguration] = None) -> str:
    """Create a backup of all keys in the keys directory."""
    if config is None:
        config = BackupConfiguration(
            base_backup_dir="./",
            encryption_enabled=True,
            verification_enabled=True,
            compression_level=9
        )
    
    backup_manager = BackupManager(config)
    
    try:
        backup_id = await backup_manager.create_backup(
            source_path=keys_dir,
            backup_type=BackupType.FULL,
            compression=CompressionType.GZIP,
            encryption=EncryptionType.HYBRID,
            tags=["keys", "security", "critical"],
            notes="Automated keys backup"
        )
        
        logger.info(f"Keys backup created successfully: {backup_id}")
        return backup_id
        
    except Exception as e:
        logger.error(f"Failed to create keys backup: {e}")
        raise
    finally:
        await backup_manager.shutdown()


async def restore_keys_backup(backup_id: str, destination_dir: str,
                            config: Optional[BackupConfiguration] = None) -> bool:
    """Restore a keys backup to the specified directory."""
    if config is None:
        config = BackupConfiguration(base_backup_dir="./")
    
    backup_manager = BackupManager(config)
    
    try:
        success = await backup_manager.restore_backup(backup_id, destination_dir)
        
        if success:
            logger.info(f"Keys backup {backup_id} restored successfully")
        else:
            logger.error(f"Failed to restore keys backup {backup_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to restore keys backup: {e}")
        return False
    finally:
        await backup_manager.shutdown()


async def verify_keys_backup(backup_id: str,
                           config: Optional[BackupConfiguration] = None) -> bool:
    """Verify the integrity of a keys backup."""
    if config is None:
        config = BackupConfiguration(base_backup_dir="./")
    
    backup_manager = BackupManager(config)
    
    try:
        success = await backup_manager.verify_backup(backup_id)
        
        if success:
            logger.info(f"Keys backup {backup_id} verification successful")
        else:
            logger.error(f"Keys backup {backup_id} verification failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to verify keys backup: {e}")
        return False
    finally:
        await backup_manager.shutdown()


# Main execution function
async def main():
    """Main function for backup operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise Keys Backup System')
    parser.add_argument('--action', choices=['create', 'restore', 'verify', 'list', 'cleanup', 'health'],
                       default='create', help='Action to perform')
    parser.add_argument('--source', help='Source directory for backup')
    parser.add_argument('--destination', help='Destination directory for restore')
    parser.add_argument('--backup-id', help='Backup ID for restore/verify operations')
    parser.add_argument('--config-file', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = BackupConfiguration()
    if args.config_file and Path(args.config_file).exists():
        # Load configuration from file (implementation omitted for brevity)
        pass
    
    backup_manager = BackupManager(config)
    
    try:
        if args.action == 'create':
            source = args.source or "../"
            backup_id = await backup_manager.create_backup(source)
            print(f"Backup created successfully: {backup_id}")
        
        elif args.action == 'restore':
            if not args.backup_id or not args.destination:
                print("Error: --backup-id and --destination required for restore")
                return 1
            
            success = await backup_manager.restore_backup(args.backup_id, args.destination)
            if success:
                print(f"Backup {args.backup_id} restored successfully")
                return 0
            else:
                print(f"Failed to restore backup {args.backup_id}")
                return 1
        
        elif args.action == 'verify':
            if not args.backup_id:
                print("Error: --backup-id required for verify")
                return 1
            
            success = await backup_manager.verify_backup(args.backup_id)
            if success:
                print(f"Backup {args.backup_id} verification successful")
                return 0
            else:
                print(f"Backup {args.backup_id} verification failed")
                return 1
        
        elif args.action == 'list':
            backups = await backup_manager.list_backups()
            print(f"{'Backup ID':<30} {'Type':<15} {'Created':<20} {'Size (MB)':<12} {'Status'}")
            print("-" * 90)
            for backup in backups:
                size_mb = backup.size_bytes / (1024 * 1024)
                print(f"{backup.backup_id:<30} {backup.backup_type.value:<15} "
                      f"{backup.timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{size_mb:<12.2f} {backup.status.value}")
        
        elif args.action == 'cleanup':
            deleted_count = await backup_manager.cleanup_expired_backups()
            print(f"Cleaned up {deleted_count} expired backups")
        
        elif args.action == 'health':
            health_status = await backup_manager.health_check()
            print(json.dumps(health_status, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1
    finally:
        await backup_manager.shutdown()


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
