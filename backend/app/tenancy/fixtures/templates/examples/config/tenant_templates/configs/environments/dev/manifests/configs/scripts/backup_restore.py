#!/usr/bin/env python3
"""
Configuration Backup & Restore System
====================================

Système avancé de sauvegarde et restauration des configurations.
Gère la synchronisation avec des systèmes de stockage externes.

Author: Backup & Recovery Team - Spotify AI Agent
Team: Infrastructure & Data Protection Division
Version: 2.0.0
Date: July 17, 2025

Usage:
    python backup_restore.py [options]
    
Examples:
    python backup_restore.py --create-backup --description "Pre-deployment backup"
    python backup_restore.py --restore --backup-id backup-20250717-143022
    python backup_restore.py --sync-to-s3 --bucket spotify-ai-backups
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import shutil
import tarfile
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import tempfile
import threading
import time

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class BackupMetadata:
    """Métadonnées complètes d'une sauvegarde."""
    backup_id: str
    timestamp: datetime
    namespace: str
    description: str
    created_by: str
    backup_type: str  # full, incremental, differential
    size_bytes: int
    compression_ratio: float
    checksum: str
    resource_count: int
    resource_types: Dict[str, int]
    cluster_info: Dict[str, str]
    retention_policy: str
    storage_locations: List[str]
    encryption_method: str
    backup_status: str  # creating, completed, failed, corrupted
    restoration_tested: bool

@dataclass
class RestoreOperation:
    """Informations d'une opération de restauration."""
    restore_id: str
    backup_id: str
    timestamp: datetime
    target_namespace: str
    restore_type: str  # full, selective, dry_run
    requested_by: str
    status: str  # running, completed, failed, cancelled
    progress_percent: int
    estimated_completion: Optional[datetime]
    restored_resources: List[str]
    failed_resources: List[str]
    validation_results: Dict[str, bool]

class BackupRestoreManager:
    """Gestionnaire avancé de sauvegarde et restauration."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 kubeconfig: Optional[str] = None,
                 backup_dir: str = "/tmp/config-backups",
                 storage_config: Optional[Dict[str, Any]] = None):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.storage_config = storage_config or {}
        
        # Configuration des clients de stockage externe
        self.s3_client = None
        self.azure_client = None
        self.gcs_client = None
        self._init_storage_clients()
        
        # Historique des opérations
        self.backup_history = []
        self.restore_history = []
        
        # Configuration de chiffrement
        self.encryption_key = self._get_or_create_encryption_key()
    
    def create_full_backup(self, 
                          description: str = "",
                          created_by: str = "system",
                          retention_policy: str = "standard",
                          encrypt: bool = True,
                          compress: bool = True,
                          verify: bool = True) -> BackupMetadata:
        """Crée une sauvegarde complète."""
        print("📦 Création d'une sauvegarde complète...")
        
        backup_id = f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        backup_start = datetime.now()
        
        try:
            # Collecte des ressources
            resources = self._collect_all_resources()
            
            # Informations du cluster
            cluster_info = self._get_cluster_info()
            
            # Création de la structure de sauvegarde
            backup_structure = {
                "metadata": {
                    "backup_id": backup_id,
                    "timestamp": backup_start.isoformat(),
                    "namespace": self.namespace,
                    "description": description,
                    "created_by": created_by,
                    "backup_type": "full",
                    "cluster_info": cluster_info,
                    "retention_policy": retention_policy
                },
                "resources": resources,
                "schema_version": "v2.0"
            }
            
            # Sauvegarde sur disque
            backup_file = self._save_backup_to_disk(backup_structure, backup_id, compress, encrypt)
            
            # Calcul des métadonnées
            backup_size = backup_file.stat().st_size
            checksum = self._calculate_file_checksum(backup_file)
            resource_types = self._count_resource_types(resources)
            
            # Métadonnées complètes
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=backup_start,
                namespace=self.namespace,
                description=description,
                created_by=created_by,
                backup_type="full",
                size_bytes=backup_size,
                compression_ratio=self._calculate_compression_ratio(backup_structure, backup_file),
                checksum=checksum,
                resource_count=len(resources),
                resource_types=resource_types,
                cluster_info=cluster_info,
                retention_policy=retention_policy,
                storage_locations=[str(backup_file)],
                encryption_method="AES-256" if encrypt else "none",
                backup_status="completed",
                restoration_tested=False
            )
            
            # Sauvegarde des métadonnées
            self._save_backup_metadata(metadata)
            
            # Vérification de l'intégrité
            if verify:
                if self._verify_backup_integrity(backup_file, metadata):
                    print("✅ Vérification d'intégrité réussie")
                else:
                    print("❌ Échec de la vérification d'intégrité")
                    metadata.backup_status = "corrupted"
            
            # Synchronisation avec le stockage externe
            self._sync_to_external_storage(backup_file, metadata)
            
            # Ajout à l'historique
            self.backup_history.append(metadata)
            
            duration = (datetime.now() - backup_start).total_seconds()
            print(f"✅ Sauvegarde créée en {duration:.1f}s")
            print(f"   ID: {backup_id}")
            print(f"   Taille: {self._format_size(backup_size)}")
            print(f"   Ressources: {len(resources)}")
            print(f"   Checksum: {checksum[:12]}...")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            # Marquer comme échouée
            failed_metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=backup_start,
                namespace=self.namespace,
                description=description,
                created_by=created_by,
                backup_type="full",
                size_bytes=0,
                compression_ratio=0.0,
                checksum="",
                resource_count=0,
                resource_types={},
                cluster_info={},
                retention_policy=retention_policy,
                storage_locations=[],
                encryption_method="none",
                backup_status="failed",
                restoration_tested=False
            )
            self.backup_history.append(failed_metadata)
            raise
    
    def create_incremental_backup(self, 
                                 base_backup_id: str,
                                 description: str = "",
                                 created_by: str = "system") -> BackupMetadata:
        """Crée une sauvegarde incrémentale."""
        print(f"📦 Création d'une sauvegarde incrémentale basée sur {base_backup_id}...")
        
        # Chargement de la sauvegarde de base
        base_backup = self._load_backup_metadata(base_backup_id)
        if not base_backup:
            raise ValueError(f"Sauvegarde de base {base_backup_id} non trouvée")
        
        base_resources = self._load_backup_resources(base_backup_id)
        current_resources = self._collect_all_resources()
        
        # Calcul des différences
        changed_resources = self._calculate_resource_differences(base_resources, current_resources)
        
        backup_id = f"inc-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        backup_start = datetime.now()
        
        # Structure de sauvegarde incrémentale
        backup_structure = {
            "metadata": {
                "backup_id": backup_id,
                "timestamp": backup_start.isoformat(),
                "namespace": self.namespace,
                "description": description,
                "created_by": created_by,
                "backup_type": "incremental",
                "base_backup_id": base_backup_id,
                "cluster_info": self._get_cluster_info()
            },
            "changed_resources": changed_resources,
            "schema_version": "v2.0"
        }
        
        # Sauvegarde sur disque
        backup_file = self._save_backup_to_disk(backup_structure, backup_id, True, True)
        
        # Métadonnées
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=backup_start,
            namespace=self.namespace,
            description=description,
            created_by=created_by,
            backup_type="incremental",
            size_bytes=backup_file.stat().st_size,
            compression_ratio=self._calculate_compression_ratio(backup_structure, backup_file),
            checksum=self._calculate_file_checksum(backup_file),
            resource_count=len(changed_resources),
            resource_types=self._count_resource_types(changed_resources),
            cluster_info=self._get_cluster_info(),
            retention_policy="standard",
            storage_locations=[str(backup_file)],
            encryption_method="AES-256",
            backup_status="completed",
            restoration_tested=False
        )
        
        self._save_backup_metadata(metadata)
        self.backup_history.append(metadata)
        
        print(f"✅ Sauvegarde incrémentale créée: {len(changed_resources)} changements")
        return metadata
    
    def restore_from_backup(self, 
                           backup_id: str,
                           target_namespace: Optional[str] = None,
                           restore_type: str = "full",
                           dry_run: bool = False,
                           requested_by: str = "system",
                           selective_resources: Optional[List[str]] = None) -> RestoreOperation:
        """Restaure depuis une sauvegarde."""
        print(f"🔄 Restauration depuis la sauvegarde {backup_id}...")
        
        restore_id = f"restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        restore_start = datetime.now()
        target_ns = target_namespace or self.namespace
        
        # Création de l'opération de restauration
        restore_op = RestoreOperation(
            restore_id=restore_id,
            backup_id=backup_id,
            timestamp=restore_start,
            target_namespace=target_ns,
            restore_type=restore_type,
            requested_by=requested_by,
            status="running",
            progress_percent=0,
            estimated_completion=None,
            restored_resources=[],
            failed_resources=[],
            validation_results={}
        )
        
        try:
            # Chargement de la sauvegarde
            backup_metadata = self._load_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValueError(f"Sauvegarde {backup_id} non trouvée")
            
            # Vérification de l'intégrité avant restauration
            if not self._verify_backup_before_restore(backup_metadata):
                raise ValueError("Échec de la vérification d'intégrité de la sauvegarde")
            
            backup_data = self._load_backup_resources(backup_id)
            
            # Sélection des ressources à restaurer
            if restore_type == "selective" and selective_resources:
                backup_data = self._filter_resources(backup_data, selective_resources)
            
            restore_op.progress_percent = 10
            
            # Sauvegarde de sécurité avant restauration
            if not dry_run:
                security_backup = self.create_full_backup(
                    description=f"Sauvegarde de sécurité avant restauration {restore_id}",
                    created_by="restore-system"
                )
                print(f"📦 Sauvegarde de sécurité créée: {security_backup.backup_id}")
            
            restore_op.progress_percent = 20
            
            # Estimation du temps de restauration
            estimated_duration = self._estimate_restore_duration(backup_data)
            restore_op.estimated_completion = restore_start + timedelta(seconds=estimated_duration)
            
            # Restauration par étapes
            total_resources = len(backup_data)
            
            for i, resource in enumerate(backup_data):
                try:
                    if dry_run:
                        # Simulation
                        self._simulate_resource_restore(resource, target_ns)
                    else:
                        # Restauration réelle
                        self._restore_resource(resource, target_ns)
                    
                    restore_op.restored_resources.append(self._get_resource_key(resource))
                    
                except Exception as e:
                    error_msg = f"Erreur lors de la restauration de {self._get_resource_key(resource)}: {e}"
                    print(f"⚠️ {error_msg}")
                    restore_op.failed_resources.append(error_msg)
                
                # Mise à jour du progrès
                restore_op.progress_percent = 20 + int((i + 1) / total_resources * 60)
            
            restore_op.progress_percent = 80
            
            # Validation post-restauration
            if not dry_run:
                restore_op.validation_results = self._validate_restoration(target_ns, backup_data)
            
            restore_op.progress_percent = 90
            
            # Finalisation
            if len(restore_op.failed_resources) == 0:
                restore_op.status = "completed"
                print("✅ Restauration terminée avec succès")
            else:
                restore_op.status = "completed_with_errors"
                print(f"⚠️ Restauration terminée avec {len(restore_op.failed_resources)} erreurs")
            
            restore_op.progress_percent = 100
            
        except Exception as e:
            print(f"❌ Erreur lors de la restauration: {e}")
            restore_op.status = "failed"
            restore_op.failed_resources.append(str(e))
        
        # Ajout à l'historique
        self.restore_history.append(restore_op)
        
        return restore_op
    
    def list_backups(self, 
                    limit: Optional[int] = None,
                    backup_type: Optional[str] = None,
                    status: Optional[str] = None) -> List[BackupMetadata]:
        """Liste les sauvegardes disponibles."""
        backups = []
        
        # Chargement depuis les métadonnées sauvegardées
        metadata_files = list(self.backup_dir.glob("*.metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                
                # Reconstruction de l'objet BackupMetadata
                metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                metadata = BackupMetadata(**metadata_dict)
                
                # Filtrage
                if backup_type and metadata.backup_type != backup_type:
                    continue
                if status and metadata.backup_status != status:
                    continue
                
                backups.append(metadata)
                
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {metadata_file}: {e}")
        
        # Tri par timestamp décroissant
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limitation du nombre de résultats
        if limit:
            backups = backups[:limit]
        
        return backups
    
    def cleanup_old_backups(self, 
                           retention_days: int = 30,
                           max_backups: int = 100,
                           dry_run: bool = False) -> Dict[str, int]:
        """Nettoie les anciennes sauvegardes selon les politiques de rétention."""
        print(f"🧹 Nettoyage des sauvegardes (rétention: {retention_days} jours, max: {max_backups})")
        
        backups = self.list_backups()
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        to_delete = []
        
        # Filtrage par âge
        for backup in backups:
            if backup.timestamp < cutoff_date:
                to_delete.append(backup)
        
        # Filtrage par nombre maximum (garder les plus récents)
        if len(backups) > max_backups:
            excess_backups = backups[max_backups:]
            to_delete.extend(excess_backups)
        
        # Suppression des doublons
        to_delete = list(set(to_delete))
        
        # Garder toujours au moins une sauvegarde
        if len(backups) - len(to_delete) < 1:
            to_delete = to_delete[:-1] if to_delete else []
        
        cleanup_stats = {
            "total_backups": len(backups),
            "deleted_count": 0,
            "freed_bytes": 0,
            "errors": 0
        }
        
        for backup in to_delete:
            try:
                if not dry_run:
                    self._delete_backup(backup)
                
                cleanup_stats["deleted_count"] += 1
                cleanup_stats["freed_bytes"] += backup.size_bytes
                
                print(f"🗑️ {'[DRY-RUN] ' if dry_run else ''}Supprimé: {backup.backup_id}")
                
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {backup.backup_id}: {e}")
                cleanup_stats["errors"] += 1
        
        print(f"✅ Nettoyage terminé: {cleanup_stats['deleted_count']} sauvegardes supprimées")
        print(f"   Espace libéré: {self._format_size(cleanup_stats['freed_bytes'])}")
        
        return cleanup_stats
    
    def sync_to_cloud_storage(self, 
                             provider: str,
                             backup_ids: Optional[List[str]] = None,
                             parallel_uploads: int = 3) -> Dict[str, str]:
        """Synchronise les sauvegardes vers le stockage cloud."""
        print(f"☁️ Synchronisation vers {provider}...")
        
        backups_to_sync = []
        
        if backup_ids:
            for backup_id in backup_ids:
                metadata = self._load_backup_metadata(backup_id)
                if metadata:
                    backups_to_sync.append(metadata)
        else:
            # Synchroniser toutes les sauvegardes non synchronisées
            backups_to_sync = [b for b in self.list_backups() if provider not in b.storage_locations]
        
        sync_results = {}
        
        # Synchronisation parallèle
        def sync_worker(backup_metadata: BackupMetadata) -> None:
            try:
                remote_url = self._upload_to_cloud(backup_metadata, provider)
                sync_results[backup_metadata.backup_id] = f"success:{remote_url}"
                
                # Mise à jour des métadonnées
                backup_metadata.storage_locations.append(remote_url)
                self._save_backup_metadata(backup_metadata)
                
            except Exception as e:
                sync_results[backup_metadata.backup_id] = f"error:{e}"
        
        # Exécution en parallèle
        threads = []
        for i in range(0, len(backups_to_sync), parallel_uploads):
            batch = backups_to_sync[i:i + parallel_uploads]
            
            for backup in batch:
                thread = threading.Thread(target=sync_worker, args=(backup,))
                threads.append(thread)
                thread.start()
            
            # Attendre la fin du batch
            for thread in threads[-len(batch):]:
                thread.join()
        
        # Résumé
        successful = sum(1 for result in sync_results.values() if result.startswith("success"))
        failed = len(sync_results) - successful
        
        print(f"✅ Synchronisation terminée: {successful} réussies, {failed} échouées")
        
        return sync_results
    
    def verify_backup_integrity(self, backup_id: str) -> bool:
        """Vérifie l'intégrité d'une sauvegarde."""
        print(f"🔍 Vérification de l'intégrité de {backup_id}...")
        
        metadata = self._load_backup_metadata(backup_id)
        if not metadata:
            print(f"❌ Métadonnées de {backup_id} non trouvées")
            return False
        
        # Vérification de l'existence du fichier
        backup_file = self._get_backup_file_path(backup_id)
        if not backup_file.exists():
            print(f"❌ Fichier de sauvegarde {backup_file} non trouvé")
            return False
        
        # Vérification du checksum
        current_checksum = self._calculate_file_checksum(backup_file)
        if current_checksum != metadata.checksum:
            print(f"❌ Checksum invalide: attendu {metadata.checksum}, obtenu {current_checksum}")
            return False
        
        # Vérification de la structure
        try:
            backup_data = self._load_backup_from_file(backup_file)
            if not self._validate_backup_structure(backup_data):
                print("❌ Structure de sauvegarde invalide")
                return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
        
        print("✅ Intégrité vérifiée")
        return True
    
    def test_restore(self, backup_id: str) -> bool:
        """Teste la restauration d'une sauvegarde en mode dry-run."""
        print(f"🧪 Test de restauration de {backup_id}...")
        
        try:
            restore_op = self.restore_from_backup(
                backup_id=backup_id,
                restore_type="full",
                dry_run=True,
                requested_by="test-system"
            )
            
            success = restore_op.status == "completed"
            
            if success:
                print("✅ Test de restauration réussi")
                
                # Marquer comme testé
                metadata = self._load_backup_metadata(backup_id)
                if metadata:
                    metadata.restoration_tested = True
                    self._save_backup_metadata(metadata)
            else:
                print(f"❌ Test de restauration échoué: {len(restore_op.failed_resources)} erreurs")
            
            return success
            
        except Exception as e:
            print(f"❌ Erreur lors du test: {e}")
            return False
    
    # Méthodes privées helper
    
    def _init_storage_clients(self) -> None:
        """Initialise les clients de stockage externe."""
        if "aws" in self.storage_config:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.storage_config["aws"].get("access_key"),
                    aws_secret_access_key=self.storage_config["aws"].get("secret_key"),
                    region_name=self.storage_config["aws"].get("region")
                )
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser le client S3: {e}")
        
        if "azure" in self.storage_config:
            try:
                self.azure_client = BlobServiceClient(
                    account_url=self.storage_config["azure"].get("account_url"),
                    credential=self.storage_config["azure"].get("credential")
                )
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser le client Azure: {e}")
        
        if "gcp" in self.storage_config:
            try:
                self.gcs_client = gcs.Client(
                    project=self.storage_config["gcp"].get("project_id")
                )
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser le client GCS: {e}")
    
    def _get_or_create_encryption_key(self) -> str:
        """Obtient ou crée une clé de chiffrement."""
        key_file = self.backup_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'r') as f:
                return f.read().strip()
        else:
            # Génération d'une nouvelle clé
            import secrets
            key = secrets.token_hex(32)
            
            with open(key_file, 'w') as f:
                f.write(key)
            
            # Protection du fichier
            key_file.chmod(0o600)
            
            return key
    
    def _collect_all_resources(self) -> List[Dict[str, Any]]:
        """Collecte toutes les ressources du namespace."""
        resources = []
        
        resource_types = [
            "pods", "deployments", "services", "configmaps", "secrets",
            "ingresses", "persistentvolumeclaims", "networkpolicies",
            "roles", "rolebindings", "serviceaccounts"
        ]
        
        for resource_type in resource_types:
            try:
                cmd = ["kubectl", "get", resource_type, "-n", self.namespace, "-o", "json"]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                for item in data.get("items", []):
                    # Nettoyage des métadonnées système
                    self._clean_resource_for_backup(item)
                    resources.append(item)
                    
            except subprocess.CalledProcessError:
                # Ressource non disponible
                continue
        
        return resources
    
    def _clean_resource_for_backup(self, resource: Dict[str, Any]) -> None:
        """Nettoie une ressource pour la sauvegarde."""
        metadata = resource.get("metadata", {})
        
        # Suppression des champs système
        system_fields = [
            "uid", "resourceVersion", "generation", "creationTimestamp",
            "managedFields", "selfLink", "finalizers"
        ]
        
        for field in system_fields:
            metadata.pop(field, None)
        
        # Suppression du statut
        resource.pop("status", None)
        
        # Nettoyage des annotations système
        annotations = metadata.get("annotations", {})
        system_annotations = [key for key in annotations.keys() 
                             if key.startswith("kubectl.kubernetes.io/")]
        
        for annotation in system_annotations:
            annotations.pop(annotation, None)
    
    def _get_cluster_info(self) -> Dict[str, str]:
        """Récupère les informations du cluster."""
        cluster_info = {}
        
        try:
            # Version de Kubernetes
            result = subprocess.run(
                ["kubectl", "version", "--short"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                cluster_info["kubernetes_version"] = result.stdout.strip()
        except Exception:
            pass
        
        try:
            # Informations du cluster
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                cluster_info["cluster_info"] = result.stdout.strip()
        except Exception:
            pass
        
        cluster_info["backup_agent_version"] = "2.0.0"
        cluster_info["backup_timestamp"] = datetime.now().isoformat()
        
        return cluster_info
    
    def _save_backup_to_disk(self, backup_structure: Dict[str, Any], 
                           backup_id: str, compress: bool, encrypt: bool) -> Path:
        """Sauvegarde la structure sur disque."""
        backup_file = self.backup_dir / f"{backup_id}.backup"
        
        # Sérialisation
        backup_content = json.dumps(backup_structure, indent=2, default=str).encode('utf-8')
        
        # Chiffrement
        if encrypt:
            backup_content = self._encrypt_data(backup_content)
            backup_file = backup_file.with_suffix('.backup.enc')
        
        # Compression
        if compress:
            backup_content = gzip.compress(backup_content)
            backup_file = backup_file.with_suffix(backup_file.suffix + '.gz')
        
        # Écriture
        with open(backup_file, 'wb') as f:
            f.write(backup_content)
        
        return backup_file
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Chiffre des données."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Génération d'une clé Fernet à partir de notre clé
            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32].ljust(32, b'0'))
            f = Fernet(key)
            
            return f.encrypt(data)
        except ImportError:
            print("⚠️ Module cryptography non disponible, chiffrement ignoré")
            return data
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Déchiffre des données."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32].ljust(32, b'0'))
            f = Fernet(key)
            
            return f.decrypt(encrypted_data)
        except ImportError:
            print("⚠️ Module cryptography non disponible, déchiffrement ignoré")
            return encrypted_data
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calcule le checksum SHA-256 d'un fichier."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _calculate_compression_ratio(self, original_data: Dict[str, Any], compressed_file: Path) -> float:
        """Calcule le ratio de compression."""
        original_size = len(json.dumps(original_data, default=str).encode('utf-8'))
        compressed_size = compressed_file.stat().st_size
        
        if original_size == 0:
            return 0.0
        
        return compressed_size / original_size
    
    def _count_resource_types(self, resources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compte les ressources par type."""
        counts = {}
        for resource in resources:
            kind = resource.get("kind", "Unknown")
            counts[kind] = counts.get(kind, 0) + 1
        return counts
    
    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Sauvegarde les métadonnées."""
        metadata_file = self.backup_dir / f"{metadata.backup_id}.metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Charge les métadonnées d'une sauvegarde."""
        metadata_file = self.backup_dir / f"{backup_id}.metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Conversion des timestamps
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
            return BackupMetadata(**data)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des métadonnées {backup_id}: {e}")
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Formate une taille en bytes en unité lisible."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Système de sauvegarde et restauration Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python backup_restore.py --create-backup --description "Sauvegarde pré-déploiement"
  python backup_restore.py --list-backups --limit 10
  python backup_restore.py --restore --backup-id backup-20250717-143022
  python backup_restore.py --sync-to-cloud aws --bucket my-backups
        """
    )
    
    parser.add_argument(
        "--namespace", "-n",
        default="spotify-ai-agent-dev",
        help="Namespace Kubernetes"
    )
    
    parser.add_argument(
        "--kubeconfig", "-k",
        help="Chemin vers le fichier kubeconfig"
    )
    
    parser.add_argument(
        "--backup-dir",
        default="/tmp/config-backups",
        help="Répertoire des sauvegardes"
    )
    
    # Actions principales
    parser.add_argument(
        "--create-backup",
        action="store_true",
        help="Crée une nouvelle sauvegarde"
    )
    
    parser.add_argument(
        "--create-incremental",
        help="Crée une sauvegarde incrémentale (spécifier l'ID de base)"
    )
    
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="Liste les sauvegardes disponibles"
    )
    
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restaure depuis une sauvegarde"
    )
    
    parser.add_argument(
        "--verify",
        help="Vérifie l'intégrité d'une sauvegarde"
    )
    
    parser.add_argument(
        "--test-restore",
        help="Teste la restauration d'une sauvegarde"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Nettoie les anciennes sauvegardes"
    )
    
    # Paramètres
    parser.add_argument(
        "--backup-id",
        help="ID de la sauvegarde"
    )
    
    parser.add_argument(
        "--description",
        default="",
        help="Description de la sauvegarde"
    )
    
    parser.add_argument(
        "--created-by",
        default="manual",
        help="Créateur de la sauvegarde"
    )
    
    parser.add_argument(
        "--target-namespace",
        help="Namespace cible pour la restauration"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limite le nombre de résultats"
    )
    
    # Stockage cloud
    parser.add_argument(
        "--sync-to-cloud",
        choices=["aws", "azure", "gcp"],
        help="Synchronise vers le stockage cloud"
    )
    
    args = parser.parse_args()
    
    try:
        # Configuration du stockage (à adapter selon l'environnement)
        storage_config = {
            "aws": {
                "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            }
        }
        
        # Création du gestionnaire
        manager = BackupRestoreManager(
            namespace=args.namespace,
            kubeconfig=args.kubeconfig,
            backup_dir=args.backup_dir,
            storage_config=storage_config
        )
        
        if args.create_backup:
            metadata = manager.create_full_backup(
                description=args.description,
                created_by=args.created_by
            )
            print(f"✅ Sauvegarde créée: {metadata.backup_id}")
        
        elif args.create_incremental:
            metadata = manager.create_incremental_backup(
                base_backup_id=args.create_incremental,
                description=args.description,
                created_by=args.created_by
            )
            print(f"✅ Sauvegarde incrémentale créée: {metadata.backup_id}")
        
        elif args.list_backups:
            backups = manager.list_backups(limit=args.limit)
            
            if backups:
                print(f"\n📋 Sauvegardes disponibles ({len(backups)}):")
                print(f"{'ID':<25} {'Type':<12} {'Date':<20} {'Taille':<10} {'Statut':<12} {'Description'}")
                print("-" * 100)
                
                for backup in backups:
                    date_str = backup.timestamp.strftime("%Y-%m-%d %H:%M")
                    size_str = manager._format_size(backup.size_bytes)
                    description = backup.description[:30] + "..." if len(backup.description) > 30 else backup.description
                    
                    print(f"{backup.backup_id:<25} {backup.backup_type:<12} {date_str:<20} {size_str:<10} {backup.backup_status:<12} {description}")
            else:
                print("Aucune sauvegarde disponible")
        
        elif args.restore:
            if not args.backup_id:
                print("❌ ID de sauvegarde requis pour la restauration")
                sys.exit(1)
            
            restore_op = manager.restore_from_backup(
                backup_id=args.backup_id,
                target_namespace=args.target_namespace,
                dry_run=args.dry_run,
                requested_by=args.created_by
            )
            
            print(f"📊 Résultats de la restauration:")
            print(f"   Statut: {restore_op.status}")
            print(f"   Ressources restaurées: {len(restore_op.restored_resources)}")
            print(f"   Ressources échouées: {len(restore_op.failed_resources)}")
        
        elif args.verify:
            success = manager.verify_backup_integrity(args.verify)
            if not success:
                sys.exit(1)
        
        elif args.test_restore:
            success = manager.test_restore(args.test_restore)
            if not success:
                sys.exit(1)
        
        elif args.cleanup:
            stats = manager.cleanup_old_backups(dry_run=args.dry_run)
            print(f"📊 Statistiques de nettoyage:")
            print(f"   Sauvegardes supprimées: {stats['deleted_count']}")
            print(f"   Espace libéré: {manager._format_size(stats['freed_bytes'])}")
        
        elif args.sync_to_cloud:
            results = manager.sync_to_cloud_storage(args.sync_to_cloud)
            successful = sum(1 for r in results.values() if r.startswith("success"))
            print(f"☁️ Synchronisation: {successful}/{len(results)} réussies")
        
        else:
            print("Aucune action spécifiée. Utilisez --help pour voir les options.")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
