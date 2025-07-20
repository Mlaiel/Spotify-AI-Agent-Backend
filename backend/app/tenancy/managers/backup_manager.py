"""
💾 Tenant Backup Manager - Gestionnaire Sauvegarde Multi-Tenant
==============================================================

Gestionnaire avancé de sauvegarde et restauration pour l'architecture multi-tenant.
Gère les sauvegardes automatisées, la réplication et la récupération de données.

Features:
- Sauvegarde automatisée multi-tenant
- Stratégies de sauvegarde (complète, incrémentale, différentielle)
- Réplication inter-sites
- Chiffrement des sauvegardes
- Compression et déduplication
- Rétention automatique
- Restauration point-in-time
- Vérification d'intégrité
- Monitoring des sauvegardes
- Alertes en cas d'échec

Author: DBA & Data Engineer + Spécialiste Sécurité
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import os
import shutil
import gzip
import hashlib
import tarfile
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, delete, text
from fastapi import HTTPException
from pydantic import BaseModel, validator
import redis.asyncio as redis
from cryptography.fernet import Fernet
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Types de sauvegarde"""
    FULL = "full"                    # Sauvegarde complète
    INCREMENTAL = "incremental"      # Sauvegarde incrémentale
    DIFFERENTIAL = "differential"    # Sauvegarde différentielle
    SNAPSHOT = "snapshot"            # Instantané
    CONTINUOUS = "continuous"        # Sauvegarde continue
    ARCHIVE = "archive"             # Archivage long terme


class BackupStatus(str, Enum):
    """États de sauvegarde"""
    PENDING = "pending"              # En attente
    RUNNING = "running"              # En cours
    COMPLETED = "completed"          # Terminée
    FAILED = "failed"               # Échouée
    VERIFYING = "verifying"         # Vérification en cours
    VERIFIED = "verified"           # Vérifiée
    CORRUPTED = "corrupted"         # Corrompue
    EXPIRED = "expired"             # Expirée


class BackupDestination(str, Enum):
    """Destinations de sauvegarde"""
    LOCAL = "local"                  # Stockage local
    S3 = "s3"                       # Amazon S3
    AZURE_BLOB = "azure_blob"       # Azure Blob Storage
    GCS = "gcs"                     # Google Cloud Storage
    NFS = "nfs"                     # Network File System
    SFTP = "sftp"                   # SFTP Server


class CompressionType(str, Enum):
    """Types de compression"""
    NONE = "none"                   # Pas de compression
    GZIP = "gzip"                   # Compression gzip
    BZIP2 = "bzip2"                 # Compression bzip2
    XZ = "xz"                       # Compression xz
    LZ4 = "lz4"                     # Compression lz4


@dataclass
class BackupPolicy:
    """Politique de sauvegarde"""
    policy_id: str
    tenant_id: str
    backup_type: BackupType
    schedule: str  # Cron expression
    retention_days: int
    compression: CompressionType = CompressionType.GZIP
    encryption_enabled: bool = True
    destinations: List[BackupDestination] = field(default_factory=list)
    pre_backup_scripts: List[str] = field(default_factory=list)
    post_backup_scripts: List[str] = field(default_factory=list)
    verification_enabled: bool = True
    max_backup_size: Optional[int] = None  # En bytes
    parallel_jobs: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BackupJob:
    """Tâche de sauvegarde"""
    job_id: str
    policy_id: str
    tenant_id: str
    backup_type: BackupType
    status: BackupStatus
    backup_path: str = ""
    backup_size: int = 0
    compression_ratio: float = 0.0
    checksum: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RestorePoint:
    """Point de restauration"""
    restore_point_id: str
    tenant_id: str
    backup_job_id: str
    timestamp: datetime
    backup_type: BackupType
    data_size: int
    backup_path: str
    dependencies: List[str] = field(default_factory=list)  # Pour sauvegardes incrémentales
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RestoreJob:
    """Tâche de restauration"""
    restore_job_id: str
    tenant_id: str
    restore_point_id: str
    target_location: str
    status: str = "pending"
    progress_percentage: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BackupRequest(BaseModel):
    """Requête de sauvegarde"""
    tenant_id: str
    backup_type: BackupType
    destinations: List[BackupDestination]
    compression: CompressionType = CompressionType.GZIP
    encryption_enabled: bool = True
    include_patterns: List[str] = []
    exclude_patterns: List[str] = []
    verify_after_backup: bool = True


class RestoreRequest(BaseModel):
    """Requête de restauration"""
    tenant_id: str
    restore_point_id: str
    target_location: str
    overwrite_existing: bool = False
    restore_permissions: bool = True
    verify_after_restore: bool = True


class TenantBackupManager:
    """
    Gestionnaire de sauvegarde multi-tenant avancé.
    
    Responsabilités:
    - Gestion des politiques de sauvegarde
    - Exécution des sauvegardes automatisées
    - Compression et chiffrement
    - Vérification d'intégrité
    - Gestion de la rétention
    - Restauration point-in-time
    - Réplication multi-sites
    """

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self.backup_policies: Dict[str, BackupPolicy] = {}
        self.active_backup_jobs: Dict[str, BackupJob] = {}
        self.restore_points: Dict[str, List[RestorePoint]] = {}
        self.active_restore_jobs: Dict[str, RestoreJob] = {}
        
        # Configuration par défaut
        self.config = {
            "backup_root_path": "/backups",
            "temp_path": "/tmp/backups",
            "max_parallel_backups": 3,
            "max_parallel_restores": 2,
            "compression_level": 6,
            "encryption_key": self._get_or_generate_encryption_key(),
            "verification_sample_size": 1024 * 1024,  # 1MB
            "default_retention_days": 30,
            "chunk_size": 64 * 1024 * 1024,  # 64MB
        }
        
        # Clients cloud (initialisés à la demande)
        self._s3_client = None
        self._azure_client = None
        self._gcs_client = None

    async def get_redis_client(self) -> redis.Redis:
        """Obtenir le client Redis"""
        if not self._redis_client:
            self._redis_client = await get_redis_client()
        return self._redis_client

    def _get_or_generate_encryption_key(self) -> str:
        """Obtenir ou générer la clé de chiffrement"""
        # En production, utiliser un gestionnaire de clés sécurisé
        key_path = "/etc/backup/encryption.key"
        try:
            if os.path.exists(key_path):
                with open(key_path, 'rb') as f:
                    return f.read()
            else:
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(key)
                return key
        except Exception:
            # Fallback vers une clé en mémoire
            return Fernet.generate_key()

    async def create_backup_policy(
        self,
        tenant_id: str,
        backup_type: BackupType,
        schedule: str,
        retention_days: int,
        destinations: List[BackupDestination],
        **kwargs
    ) -> str:
        """
        Créer une politique de sauvegarde.
        
        Args:
            tenant_id: ID du tenant
            backup_type: Type de sauvegarde
            schedule: Expression cron pour la planification
            retention_days: Nombre de jours de rétention
            destinations: Destinations de sauvegarde
            **kwargs: Options supplémentaires
            
        Returns:
            ID de la politique créée
        """
        try:
            policy_id = str(uuid.uuid4())
            
            policy = BackupPolicy(
                policy_id=policy_id,
                tenant_id=tenant_id,
                backup_type=backup_type,
                schedule=schedule,
                retention_days=retention_days,
                destinations=destinations,
                compression=kwargs.get("compression", CompressionType.GZIP),
                encryption_enabled=kwargs.get("encryption_enabled", True),
                verification_enabled=kwargs.get("verification_enabled", True),
                parallel_jobs=kwargs.get("parallel_jobs", 1)
            )

            self.backup_policies[policy_id] = policy
            await self._store_backup_policy(policy)

            # Planification de la sauvegarde
            await self._schedule_backup_policy(policy)

            logger.info(f"Politique de sauvegarde créée: {policy_id} pour tenant {tenant_id}")
            return policy_id

        except Exception as e:
            logger.error(f"Erreur création politique sauvegarde: {str(e)}")
            raise

    async def execute_backup(
        self,
        request: BackupRequest
    ) -> str:
        """
        Exécuter une sauvegarde immédiate.
        
        Args:
            request: Requête de sauvegarde
            
        Returns:
            ID de la tâche de sauvegarde
        """
        try:
            job_id = str(uuid.uuid4())
            
            # Validation des prérequis
            await self._validate_backup_prerequisites(request)

            job = BackupJob(
                job_id=job_id,
                policy_id="",  # Sauvegarde ponctuelle
                tenant_id=request.tenant_id,
                backup_type=request.backup_type,
                status=BackupStatus.PENDING
            )

            self.active_backup_jobs[job_id] = job

            # Exécution asynchrone
            asyncio.create_task(self._execute_backup_async(job, request))

            logger.info(f"Sauvegarde démarrée: {job_id} pour tenant {request.tenant_id}")
            return job_id

        except Exception as e:
            logger.error(f"Erreur démarrage sauvegarde: {str(e)}")
            raise

    async def get_backup_status(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Obtenir le statut d'une sauvegarde.
        
        Args:
            job_id: ID de la tâche
            
        Returns:
            Statut détaillé de la sauvegarde
        """
        try:
            if job_id not in self.active_backup_jobs:
                return await self._get_historical_backup_status(job_id)

            job = self.active_backup_jobs[job_id]
            
            return {
                "job_id": job_id,
                "tenant_id": job.tenant_id,
                "backup_type": job.backup_type,
                "status": job.status,
                "backup_path": job.backup_path,
                "backup_size": job.backup_size,
                "compression_ratio": job.compression_ratio,
                "timing": {
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "elapsed_time": self._calculate_elapsed_time(job)
                },
                "checksum": job.checksum,
                "metadata": job.metadata,
                "error": job.error_message,
                "logs": job.logs[-10:]  # Derniers 10 logs
            }

        except Exception as e:
            logger.error(f"Erreur récupération statut sauvegarde: {str(e)}")
            raise

    async def list_restore_points(
        self,
        tenant_id: str,
        backup_type: Optional[BackupType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Lister les points de restauration disponibles.
        
        Args:
            tenant_id: ID du tenant
            backup_type: Type de sauvegarde (optionnel)
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Liste des points de restauration
        """
        try:
            restore_points = self.restore_points.get(tenant_id, [])
            
            # Filtrage
            filtered_points = []
            for point in restore_points:
                if backup_type and point.backup_type != backup_type:
                    continue
                if start_date and point.timestamp < start_date:
                    continue
                if end_date and point.timestamp > end_date:
                    continue
                
                filtered_points.append({
                    "restore_point_id": point.restore_point_id,
                    "timestamp": point.timestamp.isoformat(),
                    "backup_type": point.backup_type,
                    "data_size": point.data_size,
                    "backup_path": point.backup_path,
                    "verified": point.verified,
                    "dependencies": point.dependencies,
                    "metadata": point.metadata
                })

            # Tri par timestamp décroissant
            filtered_points.sort(key=lambda x: x["timestamp"], reverse=True)

            return filtered_points

        except Exception as e:
            logger.error(f"Erreur liste points restauration: {str(e)}")
            return []

    async def restore_from_backup(
        self,
        request: RestoreRequest
    ) -> str:
        """
        Restaurer depuis une sauvegarde.
        
        Args:
            request: Requête de restauration
            
        Returns:
            ID de la tâche de restauration
        """
        try:
            restore_job_id = str(uuid.uuid4())
            
            # Validation du point de restauration
            restore_point = await self._find_restore_point(request.restore_point_id)
            if not restore_point:
                raise HTTPException(status_code=404, detail="Point de restauration non trouvé")

            # Validation des prérequis
            await self._validate_restore_prerequisites(request, restore_point)

            restore_job = RestoreJob(
                restore_job_id=restore_job_id,
                tenant_id=request.tenant_id,
                restore_point_id=request.restore_point_id,
                target_location=request.target_location,
                status="pending"
            )

            self.active_restore_jobs[restore_job_id] = restore_job

            # Exécution asynchrone
            asyncio.create_task(self._execute_restore_async(restore_job, request, restore_point))

            logger.info(f"Restauration démarrée: {restore_job_id} pour tenant {request.tenant_id}")
            return restore_job_id

        except Exception as e:
            logger.error(f"Erreur démarrage restauration: {str(e)}")
            raise

    async def verify_backup_integrity(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Vérifier l'intégrité d'une sauvegarde.
        
        Args:
            job_id: ID de la tâche de sauvegarde
            
        Returns:
            Rapport de vérification
        """
        try:
            job = await self._find_backup_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Sauvegarde non trouvée")

            verification_report = {
                "job_id": job_id,
                "tenant_id": job.tenant_id,
                "backup_path": job.backup_path,
                "original_checksum": job.checksum,
                "verification_timestamp": datetime.utcnow().isoformat(),
                "integrity_status": "unknown",
                "verification_details": {}
            }

            # Vérification du checksum
            if os.path.exists(job.backup_path):
                current_checksum = await self._calculate_file_checksum(job.backup_path)
                checksum_valid = current_checksum == job.checksum
                
                verification_report.update({
                    "current_checksum": current_checksum,
                    "checksum_valid": checksum_valid,
                    "integrity_status": "valid" if checksum_valid else "corrupted"
                })

                # Vérification supplémentaire du contenu
                if checksum_valid:
                    content_verification = await self._verify_backup_content(job.backup_path)
                    verification_report["verification_details"] = content_verification
                    
                    if not content_verification.get("content_valid", True):
                        verification_report["integrity_status"] = "corrupted"
            else:
                verification_report["integrity_status"] = "missing"

            return verification_report

        except Exception as e:
            logger.error(f"Erreur vérification intégrité: {str(e)}")
            return {"job_id": job_id, "integrity_status": "error", "error": str(e)}

    async def cleanup_expired_backups(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Nettoyer les sauvegardes expirées.
        
        Args:
            tenant_id: ID du tenant (tous si non spécifié)
            
        Returns:
            Statistiques de nettoyage
        """
        try:
            stats = {
                "deleted_backups": 0,
                "freed_space": 0,
                "errors": 0
            }

            # Recherche des sauvegardes expirées
            expired_backups = await self._find_expired_backups(tenant_id)

            for backup in expired_backups:
                try:
                    # Suppression du fichier
                    if os.path.exists(backup.backup_path):
                        file_size = os.path.getsize(backup.backup_path)
                        os.remove(backup.backup_path)
                        stats["freed_space"] += file_size

                    # Suppression des métadonnées
                    await self._delete_backup_metadata(backup.job_id)
                    stats["deleted_backups"] += 1

                except Exception as e:
                    logger.error(f"Erreur suppression sauvegarde {backup.job_id}: {str(e)}")
                    stats["errors"] += 1

            logger.info(f"Nettoyage terminé: {stats['deleted_backups']} sauvegardes supprimées")
            return stats

        except Exception as e:
            logger.error(f"Erreur nettoyage sauvegardes: {str(e)}")
            return {"deleted_backups": 0, "freed_space": 0, "errors": 1}

    # Méthodes privées

    async def _execute_backup_async(
        self,
        job: BackupJob,
        request: BackupRequest
    ):
        """Exécuter la sauvegarde de manière asynchrone"""
        try:
            job.status = BackupStatus.RUNNING
            job.started_at = datetime.utcnow()

            # Génération du chemin de sauvegarde
            backup_filename = self._generate_backup_filename(job, request)
            job.backup_path = os.path.join(
                self.config["backup_root_path"],
                request.tenant_id,
                backup_filename
            )

            # Création du répertoire de destination
            os.makedirs(os.path.dirname(job.backup_path), exist_ok=True)

            # Collecte des données à sauvegarder
            await self._collect_backup_data(job, request)

            # Compression si activée
            if request.compression != CompressionType.NONE:
                await self._compress_backup(job, request.compression)

            # Chiffrement si activé
            if request.encryption_enabled:
                await self._encrypt_backup(job)

            # Calcul du checksum
            job.checksum = await self._calculate_file_checksum(job.backup_path)
            job.backup_size = os.path.getsize(job.backup_path)

            # Upload vers destinations externes
            for destination in request.destinations:
                await self._upload_to_destination(job, destination)

            # Vérification si demandée
            if request.verify_after_backup:
                job.status = BackupStatus.VERIFYING
                verification_result = await self.verify_backup_integrity(job.job_id)
                
                if verification_result["integrity_status"] == "valid":
                    job.status = BackupStatus.VERIFIED
                else:
                    job.status = BackupStatus.CORRUPTED
                    job.error_message = "Échec vérification intégrité"
            else:
                job.status = BackupStatus.COMPLETED

            job.completed_at = datetime.utcnow()

            # Création du point de restauration
            if job.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]:
                await self._create_restore_point(job)

            await self._store_backup_job(job)

        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Erreur sauvegarde: {str(e)}")

    async def _collect_backup_data(
        self,
        job: BackupJob,
        request: BackupRequest
    ):
        """Collecter les données à sauvegarder"""
        # En production, collecter depuis la base de données, fichiers, etc.
        
        # Sauvegarde de la base de données
        if request.backup_type in [BackupType.FULL, BackupType.INCREMENTAL]:
            await self._backup_database(job, request)
        
        # Sauvegarde des fichiers
        await self._backup_files(job, request)
        
        # Sauvegarde de la configuration
        await self._backup_configuration(job, request)

    async def _backup_database(self, job: BackupJob, request: BackupRequest):
        """Sauvegarder la base de données"""
        # Utilisation de pg_dump pour PostgreSQL
        db_backup_path = f"{job.backup_path}.sql"
        
        cmd = [
            "pg_dump",
            "-h", "localhost",
            "-p", "5432",
            "-U", f"user_{request.tenant_id}",
            "-d", f"tenant_{request.tenant_id}",
            "-f", db_backup_path
        ]
        
        # En production, exécuter la commande de façon sécurisée
        # subprocess.run(cmd, check=True)

    async def _compress_backup(self, job: BackupJob, compression: CompressionType):
        """Compresser la sauvegarde"""
        original_path = job.backup_path
        compressed_path = f"{original_path}.gz"
        
        with open(original_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Calcul du ratio de compression
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        job.compression_ratio = compressed_size / original_size
        
        # Remplacement du fichier original
        os.remove(original_path)
        job.backup_path = compressed_path

    async def _encrypt_backup(self, job: BackupJob):
        """Chiffrer la sauvegarde"""
        fernet = Fernet(self.config["encryption_key"])
        
        original_path = job.backup_path
        encrypted_path = f"{original_path}.enc"
        
        with open(original_path, 'rb') as f_in:
            with open(encrypted_path, 'wb') as f_out:
                # Chiffrement par chunks pour les gros fichiers
                while True:
                    chunk = f_in.read(self.config["chunk_size"])
                    if not chunk:
                        break
                    encrypted_chunk = fernet.encrypt(chunk)
                    f_out.write(len(encrypted_chunk).to_bytes(4, 'big'))
                    f_out.write(encrypted_chunk)
        
        os.remove(original_path)
        job.backup_path = encrypted_path

    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculer le checksum d'un fichier"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()

    def _generate_backup_filename(self, job: BackupJob, request: BackupRequest) -> str:
        """Générer le nom de fichier de sauvegarde"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{request.backup_type}_{timestamp}_{job.job_id}.backup"

    async def _store_backup_policy(self, policy: BackupPolicy):
        """Stocker une politique de sauvegarde"""
        pass

    async def _store_backup_job(self, job: BackupJob):
        """Stocker une tâche de sauvegarde"""
        pass


# Instance globale du gestionnaire de sauvegarde
tenant_backup_manager = TenantBackupManager()
