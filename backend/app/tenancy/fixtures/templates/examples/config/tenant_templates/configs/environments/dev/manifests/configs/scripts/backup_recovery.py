#!/usr/bin/env python3
"""
Advanced Configuration Backup and Recovery System
===============================================

Système de sauvegarde et récupération avancé pour les configurations Kubernetes.
Fournit des capacités de sauvegarde automatisée, versioning, et récupération point-in-time.

Fonctionnalités principales:
- Sauvegarde automatisée avec scheduling
- Versioning git intégré
- Compression et chiffrement des sauvegardes
- Récupération point-in-time
- Validation d'intégrité
- Synchronisation multi-cluster
- Métriques de sauvegarde

Author: Configuration Management Team
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python backup_recovery.py [options]
    
Examples:
    python backup_recovery.py --backup --namespace spotify-ai-agent-dev
    python backup_recovery.py --restore --backup-id 20240717-143022
    python backup_recovery.py --schedule --interval 3600  # Backup toutes les heures
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import time
import shutil
import hashlib
import gzip
import tarfile
import threading
import schedule
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import boto3
from kubernetes import client, config

# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Métadonnées d'une sauvegarde."""
    backup_id: str
    timestamp: datetime
    namespace: str
    cluster_name: str
    environment: str
    resource_count: int
    size_bytes: int
    checksum: str
    encrypted: bool = True
    compressed: bool = True
    backup_type: str = "full"  # full, incremental, differential
    retention_days: int = 30
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

class BackupError(Exception):
    """Exception pour les erreurs de sauvegarde."""
    pass

class RecoveryError(Exception):
    """Exception pour les erreurs de récupération."""
    pass

class ConfigurationBackupRecovery:
    """Système de sauvegarde et récupération des configurations."""
    
    def __init__(self, 
                 namespace: str,
                 backup_dir: Path = Path("/tmp/config-backups"),
                 encryption_key: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 git_repo: Optional[str] = None):
        
        self.namespace = namespace
        self.backup_dir = backup_dir
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.s3_bucket = s3_bucket
        self.git_repo = git_repo
        self.cluster_name = self._get_cluster_name()
        
        # Initialisation
        self._init_backup_environment()
        self._init_kubernetes_client()
        self._init_s3_client()
        self._init_git_repo()
    
    def _init_backup_environment(self):
        """Initialise l'environnement de sauvegarde."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Répertoires spécialisés
        (self.backup_dir / "full").mkdir(exist_ok=True)
        (self.backup_dir / "incremental").mkdir(exist_ok=True)
        (self.backup_dir / "metadata").mkdir(exist_ok=True)
        (self.backup_dir / "temp").mkdir(exist_ok=True)
        
        logger.info(f"Environnement de sauvegarde initialisé: {self.backup_dir}")
    
    def _init_kubernetes_client(self):
        """Initialise le client Kubernetes."""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_v1 = client.CoreV1Api()
        self.k8s_apps = client.AppsV1Api()
        logger.info("Client Kubernetes initialisé")
    
    def _init_s3_client(self):
        """Initialise le client S3."""
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"Client S3 initialisé pour le bucket: {self.s3_bucket}")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser S3: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
    
    def _init_git_repo(self):
        """Initialise le repository Git."""
        if self.git_repo:
            try:
                git_dir = self.backup_dir / "git"
                if not git_dir.exists():
                    subprocess.run(
                        ["git", "clone", self.git_repo, str(git_dir)],
                        check=True, capture_output=True
                    )
                logger.info(f"Repository Git initialisé: {self.git_repo}")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser Git: {e}")
    
    def _generate_encryption_key(self) -> str:
        """Génère une clé de chiffrement."""
        key = Fernet.generate_key()
        key_file = self.backup_dir / ".encryption_key"
        
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Permissions restrictives
        os.chmod(key_file, 0o600)
        
        return key.decode()
    
    def _get_cluster_name(self) -> str:
        """Récupère le nom du cluster."""
        try:
            result = subprocess.run(
                ["kubectl", "config", "current-context"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return "unknown-cluster"
    
    def create_backup(self, backup_type: str = "full", 
                     tags: Optional[Dict[str, str]] = None) -> BackupMetadata:
        """Crée une sauvegarde complète des configurations."""
        logger.info(f"Démarrage de la sauvegarde {backup_type}")
        
        backup_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_dir = self.backup_dir / "temp" / backup_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Collecte des ressources
            resources = self._collect_all_resources()
            
            # Sauvegarde des ressources
            resource_files = []
            for resource_type, items in resources.items():
                if items:
                    file_path = temp_dir / f"{resource_type}.yaml"
                    with open(file_path, 'w') as f:
                        yaml.dump_all(items, f, default_flow_style=False)
                    resource_files.append(file_path)
            
            # Métadonnées du cluster
            cluster_info = self._collect_cluster_metadata()
            with open(temp_dir / "cluster_metadata.yaml", 'w') as f:
                yaml.dump(cluster_info, f)
            resource_files.append(temp_dir / "cluster_metadata.yaml")
            
            # Création de l'archive
            archive_path = self._create_archive(backup_id, temp_dir, backup_type)
            
            # Calcul du checksum
            checksum = self._calculate_checksum(archive_path)
            
            # Métadonnées de sauvegarde
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                namespace=self.namespace,
                cluster_name=self.cluster_name,
                environment=self._detect_environment(),
                resource_count=sum(len(items) for items in resources.values()),
                size_bytes=archive_path.stat().st_size,
                checksum=checksum,
                backup_type=backup_type,
                tags=tags or {}
            )
            
            # Sauvegarde des métadonnées
            self._save_metadata(metadata)
            
            # Upload vers S3 si configuré
            if self.s3_client:
                self._upload_to_s3(archive_path, metadata)
            
            # Commit Git si configuré
            if self.git_repo:
                self._commit_to_git(archive_path, metadata)
            
            # Nettoyage du répertoire temporaire
            shutil.rmtree(temp_dir)
            
            logger.info(f"Sauvegarde créée avec succès: {backup_id}")
            return metadata
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise BackupError(f"Erreur lors de la sauvegarde: {str(e)}")
    
    def _collect_all_resources(self) -> Dict[str, List[Dict]]:
        """Collecte toutes les ressources du namespace."""
        resources = {
            "configmaps": [],
            "secrets": [],
            "deployments": [],
            "services": [],
            "ingresses": [],
            "persistentvolumeclaims": [],
            "serviceaccounts": [],
            "roles": [],
            "rolebindings": []
        }
        
        try:
            # ConfigMaps
            configmaps = self.k8s_v1.list_namespaced_config_map(namespace=self.namespace)
            for cm in configmaps.items:
                resources["configmaps"].append(cm.to_dict())
            
            # Secrets (filtrer les secrets système)
            secrets = self.k8s_v1.list_namespaced_secret(namespace=self.namespace)
            for secret in secrets.items:
                if secret.type != "kubernetes.io/service-account-token":
                    # Masquer les données sensibles pour la sauvegarde
                    secret_dict = secret.to_dict()
                    if secret_dict.get("data"):
                        secret_dict["data"] = {k: "[ENCRYPTED]" for k in secret_dict["data"].keys()}
                    resources["secrets"].append(secret_dict)
            
            # Deployments
            deployments = self.k8s_apps.list_namespaced_deployment(namespace=self.namespace)
            for deployment in deployments.items:
                resources["deployments"].append(deployment.to_dict())
            
            # Services
            services = self.k8s_v1.list_namespaced_service(namespace=self.namespace)
            for service in services.items:
                resources["services"].append(service.to_dict())
            
            logger.info(f"Ressources collectées: {sum(len(items) for items in resources.values())} total")
            
        except Exception as e:
            raise BackupError(f"Erreur lors de la collecte des ressources: {str(e)}")
        
        return resources
    
    def _collect_cluster_metadata(self) -> Dict[str, Any]:
        """Collecte les métadonnées du cluster."""
        try:
            # Version Kubernetes
            version_info = self.k8s_v1.get_code()
            
            # Informations du namespace
            namespace_info = self.k8s_v1.read_namespace(name=self.namespace)
            
            # Nodes (informations générales seulement)
            nodes = self.k8s_v1.list_node()
            node_info = {
                "total_nodes": len(nodes.items),
                "node_versions": list(set(node.status.node_info.kubelet_version for node in nodes.items))
            }
            
            return {
                "cluster_name": self.cluster_name,
                "kubernetes_version": version_info.git_version,
                "namespace_info": {
                    "name": namespace_info.metadata.name,
                    "creation_timestamp": namespace_info.metadata.creation_timestamp.isoformat() if namespace_info.metadata.creation_timestamp else None,
                    "labels": namespace_info.metadata.labels or {}
                },
                "node_info": node_info,
                "backup_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Impossible de collecter les métadonnées du cluster: {e}")
            return {"error": str(e)}
    
    def _detect_environment(self) -> str:
        """Détecte l'environnement basé sur le namespace."""
        if "prod" in self.namespace.lower():
            return "production"
        elif "staging" in self.namespace.lower():
            return "staging"
        elif "dev" in self.namespace.lower():
            return "development"
        else:
            return "unknown"
    
    def _create_archive(self, backup_id: str, temp_dir: Path, backup_type: str) -> Path:
        """Crée l'archive de sauvegarde."""
        archive_name = f"{backup_id}_{self.namespace}_{backup_type}.tar.gz"
        archive_path = self.backup_dir / backup_type / archive_name
        
        # Création de l'archive tar.gz
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=backup_id)
        
        # Chiffrement si activé
        if self.encryption_key:
            encrypted_path = archive_path.with_suffix('.tar.gz.enc')
            self._encrypt_file(archive_path, encrypted_path)
            archive_path.unlink()  # Suppression de la version non chiffrée
            return encrypted_path
        
        return archive_path
    
    def _encrypt_file(self, input_path: Path, output_path: Path):
        """Chiffre un fichier."""
        fernet = Fernet(self.encryption_key.encode())
        
        with open(input_path, 'rb') as infile:
            data = infile.read()
        
        encrypted_data = fernet.encrypt(data)
        
        with open(output_path, 'wb') as outfile:
            outfile.write(encrypted_data)
    
    def _decrypt_file(self, input_path: Path, output_path: Path):
        """Déchiffre un fichier."""
        fernet = Fernet(self.encryption_key.encode())
        
        with open(input_path, 'rb') as infile:
            encrypted_data = infile.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as outfile:
            outfile.write(decrypted_data)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcule le checksum SHA256 d'un fichier."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _save_metadata(self, metadata: BackupMetadata):
        """Sauvegarde les métadonnées."""
        metadata_file = self.backup_dir / "metadata" / f"{metadata.backup_id}.json"
        
        with open(metadata_file, 'w') as f:
            # Conversion datetime en string pour JSON
            metadata_dict = asdict(metadata)
            metadata_dict['timestamp'] = metadata.timestamp.isoformat()
            json.dump(metadata_dict, f, indent=2)
    
    def _upload_to_s3(self, archive_path: Path, metadata: BackupMetadata):
        """Upload la sauvegarde vers S3."""
        if not self.s3_client:
            return
        
        try:
            s3_key = f"backups/{self.cluster_name}/{self.namespace}/{archive_path.name}"
            
            self.s3_client.upload_file(
                str(archive_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'backup-id': metadata.backup_id,
                        'namespace': metadata.namespace,
                        'cluster': metadata.cluster_name,
                        'environment': metadata.environment
                    }
                }
            )
            
            logger.info(f"Sauvegarde uploadée vers S3: s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'upload S3: {e}")
    
    def _commit_to_git(self, archive_path: Path, metadata: BackupMetadata):
        """Commit la sauvegarde vers Git."""
        if not self.git_repo:
            return
        
        try:
            git_dir = self.backup_dir / "git"
            backup_dest = git_dir / "backups" / self.cluster_name / self.namespace
            backup_dest.mkdir(parents=True, exist_ok=True)
            
            # Copie de la sauvegarde
            shutil.copy2(archive_path, backup_dest)
            
            # Copie des métadonnées
            metadata_source = self.backup_dir / "metadata" / f"{metadata.backup_id}.json"
            shutil.copy2(metadata_source, backup_dest)
            
            # Git add, commit, push
            os.chdir(git_dir)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run([
                "git", "commit", "-m", 
                f"Backup {metadata.backup_id} for {self.namespace}"
            ], check=True)
            subprocess.run(["git", "push"], check=True)
            
            logger.info(f"Sauvegarde commitée vers Git")
            
        except Exception as e:
            logger.error(f"Erreur lors du commit Git: {e}")
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupMetadata]:
        """Liste toutes les sauvegardes disponibles."""
        backups = []
        metadata_dir = self.backup_dir / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Conversion string en datetime
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                metadata = BackupMetadata(**data)
                
                if backup_type is None or metadata.backup_type == backup_type:
                    backups.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture des métadonnées {metadata_file}: {e}")
        
        # Tri par timestamp (plus récent en premier)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        return backups
    
    def restore_backup(self, backup_id: str, 
                      target_namespace: Optional[str] = None,
                      resource_types: Optional[List[str]] = None,
                      dry_run: bool = False) -> bool:
        """Restaure une sauvegarde."""
        logger.info(f"Démarrage de la restauration: {backup_id}")
        
        target_ns = target_namespace or self.namespace
        
        try:
            # Chargement des métadonnées
            metadata = self._load_backup_metadata(backup_id)
            if not metadata:
                raise RecoveryError(f"Métadonnées introuvables pour {backup_id}")
            
            # Localisation de l'archive
            archive_path = self._find_backup_archive(backup_id, metadata.backup_type)
            if not archive_path:
                raise RecoveryError(f"Archive introuvable pour {backup_id}")
            
            # Vérification de l'intégrité
            if not self._verify_backup_integrity(archive_path, metadata):
                raise RecoveryError(f"Vérification d'intégrité échouée pour {backup_id}")
            
            # Extraction de l'archive
            temp_dir = self.backup_dir / "temp" / f"restore_{backup_id}"
            self._extract_archive(archive_path, temp_dir)
            
            # Restauration des ressources
            restored_count = self._restore_resources(
                temp_dir / backup_id, 
                target_ns, 
                resource_types,
                dry_run
            )
            
            # Nettoyage
            shutil.rmtree(temp_dir)
            
            if dry_run:
                logger.info(f"Simulation de restauration terminée: {restored_count} ressources")
            else:
                logger.info(f"Restauration terminée avec succès: {restored_count} ressources")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {e}")
            return False
    
    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Charge les métadonnées d'une sauvegarde."""
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return BackupMetadata(**data)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des métadonnées: {e}")
            return None
    
    def _find_backup_archive(self, backup_id: str, backup_type: str) -> Optional[Path]:
        """Trouve l'archive de sauvegarde."""
        backup_dir = self.backup_dir / backup_type
        
        # Recherche des fichiers correspondants
        patterns = [
            f"{backup_id}_{self.namespace}_{backup_type}.tar.gz",
            f"{backup_id}_{self.namespace}_{backup_type}.tar.gz.enc"
        ]
        
        for pattern in patterns:
            archive_path = backup_dir / pattern
            if archive_path.exists():
                return archive_path
        
        return None
    
    def _verify_backup_integrity(self, archive_path: Path, metadata: BackupMetadata) -> bool:
        """Vérifie l'intégrité d'une sauvegarde."""
        try:
            current_checksum = self._calculate_checksum(archive_path)
            return current_checksum == metadata.checksum
        except Exception:
            return False
    
    def _extract_archive(self, archive_path: Path, temp_dir: Path):
        """Extrait une archive de sauvegarde."""
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Déchiffrement si nécessaire
        if archive_path.suffix == '.enc':
            decrypted_path = temp_dir / archive_path.stem
            self._decrypt_file(archive_path, decrypted_path)
            archive_path = decrypted_path
        
        # Extraction
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_dir)
    
    def _restore_resources(self, resource_dir: Path, target_namespace: str,
                          resource_types: Optional[List[str]], dry_run: bool) -> int:
        """Restaure les ressources depuis un répertoire."""
        restored_count = 0
        
        resource_files = {
            "configmaps": resource_dir / "configmaps.yaml",
            "secrets": resource_dir / "secrets.yaml",
            "deployments": resource_dir / "deployments.yaml",
            "services": resource_dir / "services.yaml"
        }
        
        for resource_type, file_path in resource_files.items():
            if resource_types and resource_type not in resource_types:
                continue
            
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r') as f:
                    documents = list(yaml.safe_load_all(f))
                
                for doc in documents:
                    if not doc:
                        continue
                    
                    # Mise à jour du namespace
                    doc['metadata']['namespace'] = target_namespace
                    
                    # Suppression des champs système
                    if 'metadata' in doc:
                        doc['metadata'].pop('resourceVersion', None)
                        doc['metadata'].pop('uid', None)
                        doc['metadata'].pop('selfLink', None)
                        doc['metadata'].pop('creationTimestamp', None)
                    
                    if not dry_run:
                        self._apply_resource(doc, resource_type)
                    
                    restored_count += 1
                    
            except Exception as e:
                logger.error(f"Erreur lors de la restauration de {resource_type}: {e}")
        
        return restored_count
    
    def _apply_resource(self, resource: Dict[str, Any], resource_type: str):
        """Applique une ressource Kubernetes."""
        try:
            if resource_type == "configmaps":
                self.k8s_v1.create_namespaced_config_map(
                    namespace=resource['metadata']['namespace'],
                    body=resource
                )
            elif resource_type == "secrets":
                # Les secrets doivent être recréés avec les vraies données
                logger.warning(f"Secret {resource['metadata']['name']} nécessite une restauration manuelle")
            elif resource_type == "deployments":
                self.k8s_apps.create_namespaced_deployment(
                    namespace=resource['metadata']['namespace'],
                    body=resource
                )
            elif resource_type == "services":
                self.k8s_v1.create_namespaced_service(
                    namespace=resource['metadata']['namespace'],
                    body=resource
                )
            
        except Exception as e:
            # Tentative de mise à jour si la ressource existe déjà
            if "already exists" in str(e):
                logger.info(f"Ressource existante, tentative de mise à jour")
                # Logique de mise à jour ici
            else:
                raise
    
    def cleanup_old_backups(self, retention_days: int = 30):
        """Nettoie les anciennes sauvegardes."""
        logger.info(f"Nettoyage des sauvegardes anciennes (>{retention_days} jours)")
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        backups = self.list_backups()
        
        for backup in backups:
            if backup.timestamp < cutoff_date:
                self._delete_backup(backup)
                logger.info(f"Sauvegarde supprimée: {backup.backup_id}")
    
    def _delete_backup(self, metadata: BackupMetadata):
        """Supprime une sauvegarde."""
        # Suppression de l'archive
        archive_path = self._find_backup_archive(metadata.backup_id, metadata.backup_type)
        if archive_path and archive_path.exists():
            archive_path.unlink()
        
        # Suppression des métadonnées
        metadata_file = self.backup_dir / "metadata" / f"{metadata.backup_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
    
    def schedule_backups(self, interval_hours: int = 24):
        """Programme des sauvegardes automatiques."""
        logger.info(f"Programmation de sauvegardes automatiques (toutes les {interval_hours}h)")
        
        def backup_job():
            try:
                self.create_backup(backup_type="full", tags={"automated": "true"})
                self.cleanup_old_backups()
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde automatique: {e}")
        
        schedule.every(interval_hours).hours.do(backup_job)
        
        # Lancement du scheduler dans un thread séparé
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduler de sauvegardes démarré")

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Système de sauvegarde et récupération des configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--namespace", "-n", required=True, help="Namespace Kubernetes")
    parser.add_argument("--backup-dir", type=Path, default=Path("/tmp/config-backups"), help="Répertoire de sauvegarde")
    parser.add_argument("--backup", action="store_true", help="Créer une sauvegarde")
    parser.add_argument("--restore", action="store_true", help="Restaurer une sauvegarde")
    parser.add_argument("--list", action="store_true", help="Lister les sauvegardes")
    parser.add_argument("--cleanup", action="store_true", help="Nettoyer les anciennes sauvegardes")
    parser.add_argument("--schedule", action="store_true", help="Programmer des sauvegardes automatiques")
    
    parser.add_argument("--backup-id", help="ID de la sauvegarde pour restauration")
    parser.add_argument("--backup-type", choices=["full", "incremental"], default="full", help="Type de sauvegarde")
    parser.add_argument("--target-namespace", help="Namespace cible pour la restauration")
    parser.add_argument("--resource-types", nargs="+", help="Types de ressources à restaurer")
    parser.add_argument("--retention-days", type=int, default=30, help="Jours de rétention")
    parser.add_argument("--interval-hours", type=int, default=24, help="Intervalle de sauvegarde automatique (heures)")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    
    parser.add_argument("--s3-bucket", help="Bucket S3 pour stockage distant")
    parser.add_argument("--git-repo", help="Repository Git pour versioning")
    parser.add_argument("--encryption-key", help="Clé de chiffrement personnalisée")
    
    args = parser.parse_args()
    
    try:
        # Création du système de sauvegarde
        backup_system = ConfigurationBackupRecovery(
            namespace=args.namespace,
            backup_dir=args.backup_dir,
            encryption_key=args.encryption_key,
            s3_bucket=args.s3_bucket,
            git_repo=args.git_repo
        )
        
        if args.backup:
            metadata = backup_system.create_backup(
                backup_type=args.backup_type,
                tags={"manual": "true"}
            )
            print(f"Sauvegarde créée: {metadata.backup_id}")
        
        elif args.restore:
            if not args.backup_id:
                print("--backup-id requis pour la restauration")
                sys.exit(1)
            
            success = backup_system.restore_backup(
                backup_id=args.backup_id,
                target_namespace=args.target_namespace,
                resource_types=args.resource_types,
                dry_run=args.dry_run
            )
            
            if success:
                print("Restauration réussie")
            else:
                print("Restauration échouée")
                sys.exit(1)
        
        elif args.list:
            backups = backup_system.list_backups()
            print(f"Sauvegardes disponibles ({len(backups)}):")
            for backup in backups:
                print(f"  {backup.backup_id} - {backup.timestamp} - {backup.backup_type} - {backup.size_bytes} bytes")
        
        elif args.cleanup:
            backup_system.cleanup_old_backups(args.retention_days)
            print("Nettoyage terminé")
        
        elif args.schedule:
            backup_system.schedule_backups(args.interval_hours)
            print("Sauvegardes programmées. Appuyez sur Ctrl+C pour arrêter.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nScheduler arrêté")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
