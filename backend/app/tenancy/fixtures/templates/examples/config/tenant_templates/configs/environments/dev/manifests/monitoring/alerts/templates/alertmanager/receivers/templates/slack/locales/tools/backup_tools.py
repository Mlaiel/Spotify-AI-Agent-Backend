"""
Système de sauvegarde et outils de maintenance automatisés.

Ce module fournit un système complet de sauvegarde avec stratégies multiples,
compression, chiffrement et outils de maintenance préventive.
"""

import os
import json
import yaml
import shutil
import tarfile
import gzip
import hashlib
import sqlite3
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
from concurrent.futures import ThreadPoolExecutor, as_completed


class BackupType(str, Enum):
    """Types de sauvegarde."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(str, Enum):
    """États de sauvegarde."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class CompressionType(str, Enum):
    """Types de compression."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    LZMA = "lzma"


class BackupConfig(BaseModel):
    """Configuration de sauvegarde."""
    name: str
    source_path: str
    destination_path: str
    backup_type: BackupType = BackupType.FULL
    compression: CompressionType = CompressionType.GZIP
    encryption_key: Optional[str] = None
    retention_days: int = 30
    schedule_cron: Optional[str] = None
    include_patterns: List[str] = Field(default_factory=list)
    exclude_patterns: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BackupRecord(BaseModel):
    """Enregistrement de sauvegarde."""
    id: str
    config_name: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    source_path: str
    backup_path: str
    file_count: int = 0
    total_size: int = 0
    compressed_size: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class FileChangeTracker:
    """Suivi des modifications de fichiers."""
    
    def __init__(self, db_path: str):
        """Initialise le tracker."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_snapshots (
                    path TEXT PRIMARY KEY,
                    size INTEGER,
                    mtime REAL,
                    checksum TEXT,
                    last_backup TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def scan_directory(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """Scanne un répertoire et retourne les métadonnées des fichiers."""
        files_info = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                
                try:
                    stat = os.stat(file_path)
                    checksum = self._calculate_checksum(file_path)
                    
                    files_info[relative_path] = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'checksum': checksum,
                        'full_path': file_path
                    }
                except (OSError, IOError) as e:
                    print(f"Erreur lors du scan de {file_path}: {e}")
        
        return files_info
    
    def get_changed_files(self, directory: str, since_backup: Optional[str] = None) -> List[str]:
        """Retourne la liste des fichiers modifiés depuis la dernière sauvegarde."""
        current_files = self.scan_directory(directory)
        changed_files = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for relative_path, file_info in current_files.items():
                # Récupération de l'état précédent
                cursor.execute(
                    "SELECT size, mtime, checksum FROM file_snapshots WHERE path = ?",
                    (relative_path,)
                )
                
                previous = cursor.fetchone()
                
                if previous is None:
                    # Nouveau fichier
                    changed_files.append(relative_path)
                else:
                    prev_size, prev_mtime, prev_checksum = previous
                    
                    # Vérification des changements
                    if (file_info['size'] != prev_size or
                        file_info['mtime'] != prev_mtime or
                        file_info['checksum'] != prev_checksum):
                        changed_files.append(relative_path)
        
        return changed_files
    
    def update_snapshots(self, directory: str, backup_id: str):
        """Met à jour les snapshots après une sauvegarde."""
        current_files = self.scan_directory(directory)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for relative_path, file_info in current_files.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO file_snapshots 
                    (path, size, mtime, checksum, last_backup, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    relative_path,
                    file_info['size'],
                    file_info['mtime'],
                    file_info['checksum'],
                    backup_id,
                    datetime.now()
                ))
            
            conn.commit()
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcule le checksum MD5 d'un fichier."""
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (OSError, IOError):
            return ""


class BackupCompressor:
    """Compresseur de sauvegardes."""
    
    @staticmethod
    def compress_file(source_path: str, dest_path: str, compression: CompressionType) -> int:
        """Compresse un fichier."""
        if compression == CompressionType.NONE:
            shutil.copy2(source_path, dest_path)
            return os.path.getsize(dest_path)
        
        elif compression == CompressionType.GZIP:
            with open(source_path, 'rb') as f_in:
                with gzip.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == CompressionType.BZIP2:
            import bz2
            with open(source_path, 'rb') as f_in:
                with bz2.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == CompressionType.XZ:
            import lzma
            with open(source_path, 'rb') as f_in:
                with lzma.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        return os.path.getsize(dest_path)
    
    @staticmethod
    def create_archive(source_dir: str, archive_path: str, compression: CompressionType,
                      include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> int:
        """Crée une archive compressée."""
        
        # Détermination du mode de compression pour tarfile
        mode_map = {
            CompressionType.NONE: 'w',
            CompressionType.GZIP: 'w:gz',
            CompressionType.BZIP2: 'w:bz2',
            CompressionType.XZ: 'w:xz'
        }
        
        mode = mode_map.get(compression, 'w:gz')
        
        def should_include(path: str) -> bool:
            """Détermine si un fichier doit être inclus."""
            import fnmatch
            
            # Vérification des patterns d'exclusion
            if exclude_patterns:
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(path, pattern):
                        return False
            
            # Vérification des patterns d'inclusion
            if include_patterns:
                for pattern in include_patterns:
                    if fnmatch.fnmatch(path, pattern):
                        return True
                return False  # Si des patterns d'inclusion sont définis, exclure par défaut
            
            return True
        
        with tarfile.open(archive_path, mode) as tar:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, source_dir)
                    
                    if should_include(relative_path):
                        tar.add(file_path, arcname=relative_path)
        
        return os.path.getsize(archive_path)


class BackupEncryption:
    """Chiffrement de sauvegardes."""
    
    @staticmethod
    def generate_key() -> str:
        """Génère une clé de chiffrement."""
        return Fernet.generate_key().decode()
    
    @staticmethod
    def encrypt_file(source_path: str, dest_path: str, encryption_key: str):
        """Chiffre un fichier."""
        fernet = Fernet(encryption_key.encode())
        
        with open(source_path, 'rb') as f_in:
            data = f_in.read()
        
        encrypted_data = fernet.encrypt(data)
        
        with open(dest_path, 'wb') as f_out:
            f_out.write(encrypted_data)
    
    @staticmethod
    def decrypt_file(source_path: str, dest_path: str, encryption_key: str):
        """Déchiffre un fichier."""
        fernet = Fernet(encryption_key.encode())
        
        with open(source_path, 'rb') as f_in:
            encrypted_data = f_in.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        with open(dest_path, 'wb') as f_out:
            f_out.write(decrypted_data)


class BackupEngine:
    """Moteur de sauvegarde principal."""
    
    def __init__(self, storage_path: str):
        """Initialise le moteur de sauvegarde."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "backup_metadata.db"
        self.tracker_db = self.storage_path / "file_tracker.db"
        
        self.file_tracker = FileChangeTracker(str(self.tracker_db))
        self._init_database()
        
        self.configs: Dict[str, BackupConfig] = {}
        self.running_backups: Dict[str, asyncio.Task] = {}
    
    def _init_database(self):
        """Initialise la base de données des sauvegardes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_records (
                    id TEXT PRIMARY KEY,
                    config_name TEXT,
                    backup_type TEXT,
                    status TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    source_path TEXT,
                    backup_path TEXT,
                    file_count INTEGER,
                    total_size INTEGER,
                    compressed_size INTEGER,
                    checksum TEXT,
                    metadata TEXT,
                    error_message TEXT
                )
            """)
            conn.commit()
    
    def add_config(self, config: BackupConfig):
        """Ajoute une configuration de sauvegarde."""
        self.configs[config.name] = config
    
    def remove_config(self, config_name: str):
        """Supprime une configuration de sauvegarde."""
        if config_name in self.configs:
            del self.configs[config_name]
    
    async def create_backup(self, config_name: str) -> BackupRecord:
        """Crée une sauvegarde."""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' non trouvée")
        
        config = self.configs[config_name]
        backup_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = BackupRecord(
            id=backup_id,
            config_name=config_name,
            backup_type=config.backup_type,
            status=BackupStatus.PENDING,
            start_time=datetime.now(),
            source_path=config.source_path,
            backup_path=""
        )
        
        try:
            # Enregistrement du début de sauvegarde
            self._save_record(record)
            
            # Création de la sauvegarde
            record = await self._perform_backup(config, record)
            
            # Vérification de l'intégrité
            if await self._verify_backup(record):
                record.status = BackupStatus.VERIFIED
            else:
                record.status = BackupStatus.CORRUPTED
            
        except Exception as e:
            record.status = BackupStatus.FAILED
            record.error_message = str(e)
            record.end_time = datetime.now()
        
        finally:
            # Sauvegarde finale de l'enregistrement
            self._save_record(record)
        
        return record
    
    async def _perform_backup(self, config: BackupConfig, record: BackupRecord) -> BackupRecord:
        """Effectue la sauvegarde."""
        record.status = BackupStatus.RUNNING
        self._save_record(record)
        
        source_path = Path(config.source_path)
        backup_dir = self.storage_path / config.name / record.id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if config.backup_type == BackupType.FULL:
            files_to_backup = self._get_all_files(source_path, config)
        elif config.backup_type == BackupType.INCREMENTAL:
            files_to_backup = self._get_changed_files(source_path, config)
        else:
            files_to_backup = self._get_all_files(source_path, config)
        
        # Création de l'archive
        archive_name = f"{record.id}.tar"
        if config.compression != CompressionType.NONE:
            archive_name += f".{config.compression.value}"
        
        archive_path = backup_dir / archive_name
        
        # Compression
        compressed_size = BackupCompressor.create_archive(
            str(source_path),
            str(archive_path),
            config.compression,
            config.include_patterns,
            config.exclude_patterns
        )
        
        # Chiffrement si configuré
        if config.encryption_key:
            encrypted_path = backup_dir / f"{archive_name}.enc"
            BackupEncryption.encrypt_file(
                str(archive_path),
                str(encrypted_path),
                config.encryption_key
            )
            archive_path.unlink()  # Suppression de l'archive non chiffrée
            archive_path = encrypted_path
        
        # Calcul du checksum
        checksum = self._calculate_file_checksum(str(archive_path))
        
        # Mise à jour de l'enregistrement
        record.backup_path = str(archive_path)
        record.file_count = len(files_to_backup)
        record.total_size = sum(f.stat().st_size for f in files_to_backup if f.exists())
        record.compressed_size = compressed_size
        record.checksum = checksum
        record.end_time = datetime.now()
        record.status = BackupStatus.COMPLETED
        
        # Mise à jour du tracker de fichiers
        self.file_tracker.update_snapshots(str(source_path), record.id)
        
        return record
    
    def _get_all_files(self, source_path: Path, config: BackupConfig) -> List[Path]:
        """Récupère tous les fichiers à sauvegarder."""
        files = []
        import fnmatch
        
        for root, dirs, filenames in os.walk(source_path):
            for filename in filenames:
                file_path = Path(root) / filename
                relative_path = file_path.relative_to(source_path)
                
                # Vérification des patterns
                should_include = True
                
                # Exclusions
                for pattern in config.exclude_patterns:
                    if fnmatch.fnmatch(str(relative_path), pattern):
                        should_include = False
                        break
                
                # Inclusions (si spécifiées)
                if should_include and config.include_patterns:
                    should_include = False
                    for pattern in config.include_patterns:
                        if fnmatch.fnmatch(str(relative_path), pattern):
                            should_include = True
                            break
                
                if should_include:
                    files.append(file_path)
        
        return files
    
    def _get_changed_files(self, source_path: Path, config: BackupConfig) -> List[Path]:
        """Récupère les fichiers modifiés pour une sauvegarde incrémentale."""
        changed_files = self.file_tracker.get_changed_files(str(source_path))
        return [source_path / f for f in changed_files]
    
    async def _verify_backup(self, record: BackupRecord) -> bool:
        """Vérifie l'intégrité d'une sauvegarde."""
        try:
            # Vérification du checksum
            current_checksum = self._calculate_file_checksum(record.backup_path)
            if current_checksum != record.checksum:
                return False
            
            # Vérification de l'archive (test d'extraction)
            backup_path = Path(record.backup_path)
            
            if backup_path.suffix == '.enc':
                # Déchiffrement temporaire pour vérification
                config = self.configs[record.config_name]
                temp_path = backup_path.with_suffix('')
                BackupEncryption.decrypt_file(
                    str(backup_path),
                    str(temp_path),
                    config.encryption_key
                )
                
                # Test de l'archive
                result = self._test_archive(str(temp_path))
                temp_path.unlink()  # Nettoyage
                
                return result
            else:
                return self._test_archive(str(backup_path))
        
        except Exception as e:
            print(f"Erreur lors de la vérification: {e}")
            return False
    
    def _test_archive(self, archive_path: str) -> bool:
        """Teste l'intégrité d'une archive."""
        try:
            with tarfile.open(archive_path, 'r') as tar:
                # Test de tous les membres
                for member in tar.getmembers():
                    if member.isfile():
                        tar.extractfile(member)
            return True
        except Exception:
            return False
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calcule le checksum SHA256 d'un fichier."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _save_record(self, record: BackupRecord):
        """Sauvegarde un enregistrement en base."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backup_records 
                (id, config_name, backup_type, status, start_time, end_time,
                 source_path, backup_path, file_count, total_size, compressed_size,
                 checksum, metadata, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.config_name,
                record.backup_type.value,
                record.status.value,
                record.start_time,
                record.end_time,
                record.source_path,
                record.backup_path,
                record.file_count,
                record.total_size,
                record.compressed_size,
                record.checksum,
                json.dumps(record.metadata),
                record.error_message
            ))
            conn.commit()
    
    def get_backup_history(self, config_name: Optional[str] = None, limit: int = 100) -> List[BackupRecord]:
        """Récupère l'historique des sauvegardes."""
        with sqlite3.connect(self.db_path) as conn:
            if config_name:
                cursor = conn.execute("""
                    SELECT * FROM backup_records 
                    WHERE config_name = ?
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (config_name, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM backup_records 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (limit,))
            
            records = []
            for row in cursor.fetchall():
                record = BackupRecord(
                    id=row[0],
                    config_name=row[1],
                    backup_type=BackupType(row[2]),
                    status=BackupStatus(row[3]),
                    start_time=datetime.fromisoformat(row[4]),
                    end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                    source_path=row[6],
                    backup_path=row[7],
                    file_count=row[8],
                    total_size=row[9],
                    compressed_size=row[10],
                    checksum=row[11],
                    metadata=json.loads(row[12]) if row[12] else {},
                    error_message=row[13]
                )
                records.append(record)
            
            return records
    
    async def restore_backup(self, backup_id: str, restore_path: str) -> bool:
        """Restaure une sauvegarde."""
        # Récupération de l'enregistrement
        records = [r for r in self.get_backup_history() if r.id == backup_id]
        if not records:
            raise ValueError(f"Sauvegarde {backup_id} non trouvée")
        
        record = records[0]
        config = self.configs.get(record.config_name)
        
        if not config:
            raise ValueError(f"Configuration {record.config_name} non trouvée")
        
        try:
            backup_path = Path(record.backup_path)
            restore_path = Path(restore_path)
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Déchiffrement si nécessaire
            if backup_path.suffix == '.enc':
                temp_path = backup_path.with_suffix('')
                BackupEncryption.decrypt_file(
                    str(backup_path),
                    str(temp_path),
                    config.encryption_key
                )
                backup_path = temp_path
            
            # Extraction de l'archive
            with tarfile.open(str(backup_path), 'r') as tar:
                tar.extractall(str(restore_path))
            
            # Nettoyage si déchiffrement temporaire
            if backup_path.suffix != '.enc' and str(backup_path).endswith('.enc'):
                backup_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"Erreur lors de la restauration: {e}")
            return False
    
    def cleanup_old_backups(self, config_name: str):
        """Nettoie les anciennes sauvegardes selon la rétention."""
        config = self.configs.get(config_name)
        if not config:
            return
        
        cutoff_date = datetime.now() - timedelta(days=config.retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, backup_path FROM backup_records 
                WHERE config_name = ? AND start_time < ?
            """, (config_name, cutoff_date))
            
            for backup_id, backup_path in cursor.fetchall():
                try:
                    # Suppression du fichier
                    if backup_path and os.path.exists(backup_path):
                        os.remove(backup_path)
                    
                    # Suppression du répertoire si vide
                    backup_dir = Path(backup_path).parent
                    if backup_dir.exists() and not any(backup_dir.iterdir()):
                        backup_dir.rmdir()
                    
                    # Suppression de l'enregistrement
                    conn.execute("DELETE FROM backup_records WHERE id = ?", (backup_id,))
                    
                except Exception as e:
                    print(f"Erreur lors du nettoyage de {backup_id}: {e}")
            
            conn.commit()


class MaintenanceTask(BaseModel):
    """Tâche de maintenance."""
    name: str
    description: str
    command: Union[str, List[str]]
    schedule_cron: Optional[str] = None
    timeout: int = 3600
    retry_count: int = 3
    enabled: bool = True
    environment: Dict[str, str] = Field(default_factory=dict)


class MaintenanceEngine:
    """Moteur de maintenance automatisée."""
    
    def __init__(self):
        """Initialise le moteur de maintenance."""
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.running = False
    
    def add_task(self, task: MaintenanceTask):
        """Ajoute une tâche de maintenance."""
        self.tasks[task.name] = task
    
    def remove_task(self, task_name: str):
        """Supprime une tâche de maintenance."""
        if task_name in self.tasks:
            del self.tasks[task_name]
    
    async def run_task(self, task_name: str) -> Dict[str, Any]:
        """Exécute une tâche de maintenance."""
        if task_name not in self.tasks:
            raise ValueError(f"Tâche '{task_name}' non trouvée")
        
        task = self.tasks[task_name]
        
        if not task.enabled:
            return {
                'task_name': task_name,
                'status': 'skipped',
                'message': 'Tâche désactivée'
            }
        
        result = {
            'task_name': task_name,
            'start_time': datetime.now(),
            'status': 'running'
        }
        
        for attempt in range(task.retry_count + 1):
            try:
                # Préparation de la commande
                if isinstance(task.command, str):
                    cmd = task.command
                    shell = True
                else:
                    cmd = task.command
                    shell = False
                
                # Exécution
                process = await asyncio.create_subprocess_shell(
                    cmd if shell else ' '.join(cmd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **task.environment}
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=task.timeout
                    )
                    
                    result.update({
                        'status': 'completed' if process.returncode == 0 else 'failed',
                        'return_code': process.returncode,
                        'stdout': stdout.decode('utf-8'),
                        'stderr': stderr.decode('utf-8'),
                        'attempt': attempt + 1,
                        'end_time': datetime.now()
                    })
                    
                    if process.returncode == 0:
                        break
                    
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    result.update({
                        'status': 'timeout',
                        'message': f'Timeout après {task.timeout}s',
                        'attempt': attempt + 1,
                        'end_time': datetime.now()
                    })
                
            except Exception as e:
                result.update({
                    'status': 'error',
                    'message': str(e),
                    'attempt': attempt + 1,
                    'end_time': datetime.now()
                })
            
            # Attente avant retry
            if attempt < task.retry_count and result['status'] != 'completed':
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Enregistrement dans l'historique
        self.task_history.append(result.copy())
        
        return result
    
    async def run_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Exécute toutes les tâches de maintenance."""
        results = {}
        
        for task_name in self.tasks:
            result = await self.run_task(task_name)
            results[task_name] = result
        
        return results
    
    def get_task_history(self, task_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère l'historique des tâches."""
        history = self.task_history
        
        if task_name:
            history = [h for h in history if h['task_name'] == task_name]
        
        return sorted(history, key=lambda x: x['start_time'], reverse=True)[:limit]


# Configuration des tâches de maintenance par défaut
DEFAULT_MAINTENANCE_TASKS = [
    MaintenanceTask(
        name="cleanup_logs",
        description="Nettoyage des logs anciens",
        command="find /var/log -name '*.log' -mtime +30 -delete",
        schedule_cron="0 2 * * *"  # 2h du matin tous les jours
    ),
    MaintenanceTask(
        name="disk_cleanup",
        description="Nettoyage de l'espace disque",
        command="df -h && du -sh /tmp/* | sort -hr | head -20",
        schedule_cron="0 3 * * 0"  # 3h du matin tous les dimanches
    ),
    MaintenanceTask(
        name="check_services",
        description="Vérification des services système",
        command="systemctl status --failed",
        schedule_cron="*/30 * * * *"  # Toutes les 30 minutes
    ),
    MaintenanceTask(
        name="update_package_cache",
        description="Mise à jour du cache des paquets",
        command="apt update && apt list --upgradable",
        schedule_cron="0 6 * * *"  # 6h du matin tous les jours
    )
]


# Factory functions
def create_backup_engine(storage_path: str = "/var/backups") -> BackupEngine:
    """Crée un moteur de sauvegarde."""
    return BackupEngine(storage_path)


def create_maintenance_engine() -> MaintenanceEngine:
    """Crée un moteur de maintenance."""
    engine = MaintenanceEngine()
    
    # Ajout des tâches par défaut
    for task in DEFAULT_MAINTENANCE_TASKS:
        engine.add_task(task)
    
    return engine


async def setup_automated_backup(
    name: str,
    source_path: str,
    destination_path: str,
    backup_type: BackupType = BackupType.FULL,
    compression: CompressionType = CompressionType.GZIP,
    retention_days: int = 30
) -> BackupEngine:
    """Configure une sauvegarde automatisée."""
    engine = create_backup_engine(destination_path)
    
    config = BackupConfig(
        name=name,
        source_path=source_path,
        destination_path=destination_path,
        backup_type=backup_type,
        compression=compression,
        retention_days=retention_days
    )
    
    engine.add_config(config)
    
    return engine
