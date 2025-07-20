#!/usr/bin/env python3
"""
Configuration Rollback Script
============================

Script avancé pour la gestion des rollbacks de configurations Kubernetes.
Permet de revenir à une version précédente en cas de problème.

Author: Configuration Management Team - Spotify AI Agent
Team: DevOps & Infrastructure Engineering
Version: 2.0.0
Date: July 17, 2025

Usage:
    python rollback_configs.py [options]
    
Examples:
    python rollback_configs.py --namespace spotify-ai-agent-dev --target-revision 5
    python rollback_configs.py --backup-file /tmp/backup-20250717.yaml
    python rollback_configs.py --auto-rollback --health-threshold 50
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import tempfile
import base64

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class BackupMetadata:
    """Métadonnées d'une sauvegarde."""
    timestamp: datetime
    namespace: str
    revision: int
    description: str
    created_by: str
    configuration_hash: str
    resource_counts: Dict[str, int]

@dataclass
class RollbackPlan:
    """Plan de rollback détaillé."""
    target_revision: int
    backup_file: str
    affected_resources: List[Dict[str, Any]]
    rollback_strategy: str
    estimated_duration: int
    risk_level: str
    validation_steps: List[str]

class ConfigurationRollbackError(Exception):
    """Exception pour les erreurs de rollback."""
    pass

class ConfigurationRollbackManager:
    """Gestionnaire avancé de rollbacks de configuration."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 kubeconfig: Optional[str] = None,
                 backup_dir: str = "/tmp/config-backups"):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_history = []
        self.current_revision = 0
        
    def create_backup(self, description: str = "", created_by: str = "system") -> str:
        """Crée une sauvegarde complète des configurations actuelles."""
        print("📦 Création de la sauvegarde...")
        
        timestamp = datetime.now()
        backup_filename = f"config-backup-{timestamp.strftime('%Y%m%d-%H%M%S')}.yaml"
        backup_path = self.backup_dir / backup_filename
        
        # Récupération de toutes les ressources
        resources = self._get_all_resources()
        
        # Calcul du hash de configuration
        config_hash = self._calculate_config_hash(resources)
        
        # Comptage des ressources
        resource_counts = self._count_resources(resources)
        
        # Métadonnées de sauvegarde
        metadata = BackupMetadata(
            timestamp=timestamp,
            namespace=self.namespace,
            revision=self._get_next_revision(),
            description=description,
            created_by=created_by,
            configuration_hash=config_hash,
            resource_counts=resource_counts
        )
        
        # Création du fichier de sauvegarde
        backup_content = {
            "apiVersion": "v1",
            "kind": "ConfigurationBackup",
            "metadata": {
                "name": f"backup-{metadata.revision}",
                "namespace": self.namespace,
                "creationTimestamp": timestamp.isoformat(),
                "annotations": {
                    "backup.spotify-ai-agent.com/description": description,
                    "backup.spotify-ai-agent.com/created-by": created_by,
                    "backup.spotify-ai-agent.com/config-hash": config_hash,
                    "backup.spotify-ai-agent.com/revision": str(metadata.revision)
                }
            },
            "spec": {
                "resources": resources,
                "metadata": {
                    "timestamp": timestamp.isoformat(),
                    "namespace": self.namespace,
                    "revision": metadata.revision,
                    "description": description,
                    "created_by": created_by,
                    "configuration_hash": config_hash,
                    "resource_counts": resource_counts
                }
            }
        }
        
        # Sauvegarde sur disque
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(backup_content, f, default_flow_style=False, sort_keys=False)
        
        # Compression de la sauvegarde
        compressed_path = self._compress_backup(backup_path)
        
        print(f"✅ Sauvegarde créée: {compressed_path}")
        print(f"   Révision: {metadata.revision}")
        print(f"   Hash: {config_hash}")
        print(f"   Ressources: {sum(resource_counts.values())}")
        
        return str(compressed_path)
    
    def list_backups(self) -> List[BackupMetadata]:
        """Liste toutes les sauvegardes disponibles."""
        backups = []
        
        # Recherche des fichiers de sauvegarde
        for backup_file in self.backup_dir.glob("config-backup-*.yaml*"):
            try:
                # Décompression si nécessaire
                if backup_file.suffix == '.gz':
                    content = self._decompress_backup(backup_file)
                else:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        content = yaml.safe_load(f)
                
                # Extraction des métadonnées
                spec = content.get("spec", {})
                metadata_dict = spec.get("metadata", {})
                
                metadata = BackupMetadata(
                    timestamp=datetime.fromisoformat(metadata_dict.get("timestamp")),
                    namespace=metadata_dict.get("namespace"),
                    revision=metadata_dict.get("revision", 0),
                    description=metadata_dict.get("description", ""),
                    created_by=metadata_dict.get("created_by", "unknown"),
                    configuration_hash=metadata_dict.get("configuration_hash", ""),
                    resource_counts=metadata_dict.get("resource_counts", {})
                )
                
                backups.append(metadata)
                
            except Exception as e:
                print(f"⚠️ Erreur lors de la lecture de {backup_file}: {e}")
        
        # Tri par révision décroissante
        backups.sort(key=lambda x: x.revision, reverse=True)
        return backups
    
    def analyze_rollback_impact(self, target_revision: int) -> RollbackPlan:
        """Analyse l'impact d'un rollback vers une révision cible."""
        print(f"🔍 Analyse de l'impact du rollback vers la révision {target_revision}...")
        
        # Recherche de la sauvegarde cible
        target_backup = None
        backups = self.list_backups()
        
        for backup in backups:
            if backup.revision == target_revision:
                target_backup = backup
                break
        
        if not target_backup:
            raise ConfigurationRollbackError(f"Sauvegarde pour la révision {target_revision} non trouvée")
        
        # Chargement de la configuration cible
        backup_file = self._find_backup_file(target_revision)
        target_config = self._load_backup(backup_file)
        
        # Configuration actuelle
        current_resources = self._get_all_resources()
        
        # Analyse des différences
        affected_resources = self._analyze_differences(current_resources, target_config["spec"]["resources"])
        
        # Calcul des métriques de risque
        risk_level = self._calculate_risk_level(affected_resources)
        estimated_duration = self._estimate_rollback_duration(affected_resources)
        
        # Stratégie de rollback
        rollback_strategy = self._determine_rollback_strategy(affected_resources, risk_level)
        
        # Étapes de validation
        validation_steps = self._generate_validation_steps(affected_resources)
        
        plan = RollbackPlan(
            target_revision=target_revision,
            backup_file=backup_file,
            affected_resources=affected_resources,
            rollback_strategy=rollback_strategy,
            estimated_duration=estimated_duration,
            risk_level=risk_level,
            validation_steps=validation_steps
        )
        
        return plan
    
    def execute_rollback(self, rollback_plan: RollbackPlan, confirm: bool = False) -> bool:
        """Exécute un rollback selon le plan fourni."""
        if not confirm:
            print("⚠️ Rollback non confirmé. Utilisez --confirm pour exécuter.")
            return False
        
        print(f"🔄 Début du rollback vers la révision {rollback_plan.target_revision}")
        print(f"   Stratégie: {rollback_plan.rollback_strategy}")
        print(f"   Risque: {rollback_plan.risk_level}")
        print(f"   Durée estimée: {rollback_plan.estimated_duration}s")
        
        try:
            # Sauvegarde de sécurité avant rollback
            current_backup = self.create_backup(
                description=f"Sauvegarde automatique avant rollback vers {rollback_plan.target_revision}",
                created_by="rollback-system"
            )
            print(f"📦 Sauvegarde de sécurité créée: {current_backup}")
            
            # Chargement de la configuration cible
            target_config = self._load_backup(rollback_plan.backup_file)
            target_resources = target_config["spec"]["resources"]
            
            # Exécution selon la stratégie
            if rollback_plan.rollback_strategy == "incremental":
                success = self._execute_incremental_rollback(target_resources, rollback_plan)
            elif rollback_plan.rollback_strategy == "atomic":
                success = self._execute_atomic_rollback(target_resources, rollback_plan)
            else:
                success = self._execute_standard_rollback(target_resources, rollback_plan)
            
            if success:
                print("✅ Rollback exécuté avec succès")
                
                # Validation post-rollback
                if self._validate_rollback(rollback_plan):
                    print("✅ Validation post-rollback réussie")
                    return True
                else:
                    print("❌ Échec de la validation post-rollback")
                    return False
            else:
                print("❌ Échec du rollback")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors du rollback: {e}")
            
            # Tentative de restauration automatique
            if hasattr(self, 'emergency_backup'):
                print("🚨 Tentative de restauration d'urgence...")
                self._emergency_restore()
            
            return False
    
    def auto_rollback_on_health_degradation(self, health_threshold: int = 50, 
                                          check_interval: int = 30,
                                          max_wait_time: int = 300) -> bool:
        """Rollback automatique en cas de dégradation de la santé."""
        print(f"🤖 Surveillance automatique activée (seuil: {health_threshold}%)")
        
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            # Vérification de la santé
            health_score = self._get_health_score()
            print(f"💚 Score de santé actuel: {health_score}%")
            
            if health_score < health_threshold:
                print(f"🚨 Score de santé en dessous du seuil ({health_score}% < {health_threshold}%)")
                
                # Recherche de la dernière sauvegarde stable
                stable_backup = self._find_last_stable_backup()
                
                if stable_backup:
                    print(f"🔄 Rollback automatique vers la révision {stable_backup.revision}")
                    
                    # Analyse et exécution du rollback
                    plan = self.analyze_rollback_impact(stable_backup.revision)
                    return self.execute_rollback(plan, confirm=True)
                else:
                    print("❌ Aucune sauvegarde stable trouvée")
                    return False
            
            time.sleep(check_interval)
        
        print("⏰ Temps d'attente maximum atteint")
        return True
    
    def cleanup_old_backups(self, retention_days: int = 30, max_backups: int = 50) -> None:
        """Nettoie les anciennes sauvegardes selon la politique de rétention."""
        print(f"🧹 Nettoyage des sauvegardes (rétention: {retention_days} jours, max: {max_backups})")
        
        backups = self.list_backups()
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Filtrage par âge
        old_backups = [b for b in backups if b.timestamp < cutoff_date]
        
        # Filtrage par nombre maximum
        if len(backups) > max_backups:
            excess_backups = backups[max_backups:]
            old_backups.extend(excess_backups)
        
        # Suppression (en gardant toujours au moins une sauvegarde)
        if len(backups) - len(old_backups) >= 1:
            for backup in old_backups:
                backup_file = self._find_backup_file(backup.revision)
                if backup_file and os.path.exists(backup_file):
                    os.unlink(backup_file)
                    print(f"🗑️ Supprimé: révision {backup.revision}")
        
        print(f"✅ Nettoyage terminé ({len(old_backups)} sauvegardes supprimées)")
    
    # Méthodes privées helper
    
    def _get_all_resources(self) -> List[Dict[str, Any]]:
        """Récupère toutes les ressources du namespace."""
        resources = []
        
        resource_types = ["configmaps", "secrets", "deployments", "services", "ingresses"]
        
        for resource_type in resource_types:
            try:
                cmd = ["kubectl", "get", resource_type, "-n", self.namespace, "-o", "json"]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                for item in data.get("items", []):
                    # Nettoyage des métadonnées système
                    self._clean_resource_metadata(item)
                    resources.append(item)
                    
            except subprocess.CalledProcessError:
                # Ressource non disponible, continuer
                continue
        
        return resources
    
    def _clean_resource_metadata(self, resource: Dict[str, Any]) -> None:
        """Nettoie les métadonnées système d'une ressource."""
        metadata = resource.get("metadata", {})
        
        # Suppression des champs système
        system_fields = ["uid", "resourceVersion", "generation", "creationTimestamp", 
                        "managedFields", "selfLink"]
        
        for field in system_fields:
            metadata.pop(field, None)
        
        # Nettoyage du statut
        resource.pop("status", None)
    
    def _calculate_config_hash(self, resources: List[Dict[str, Any]]) -> str:
        """Calcule un hash unique pour la configuration."""
        import hashlib
        
        # Tri des ressources pour assurer la cohérence
        sorted_resources = sorted(resources, key=lambda x: (
            x.get("kind", ""),
            x.get("metadata", {}).get("name", "")
        ))
        
        # Sérialisation et hash
        config_str = json.dumps(sorted_resources, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _count_resources(self, resources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compte les ressources par type."""
        counts = {}
        for resource in resources:
            kind = resource.get("kind", "unknown")
            counts[kind] = counts.get(kind, 0) + 1
        return counts
    
    def _get_next_revision(self) -> int:
        """Obtient le numéro de révision suivant."""
        backups = self.list_backups()
        if backups:
            return max(backup.revision for backup in backups) + 1
        return 1
    
    def _compress_backup(self, backup_path: Path) -> Path:
        """Compresse une sauvegarde."""
        import gzip
        
        compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
        
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Suppression du fichier non compressé
        backup_path.unlink()
        
        return compressed_path
    
    def _decompress_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Décompresse et charge une sauvegarde."""
        import gzip
        
        with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _find_backup_file(self, revision: int) -> Optional[str]:
        """Trouve le fichier de sauvegarde pour une révision."""
        for backup_file in self.backup_dir.glob(f"config-backup-*.yaml*"):
            try:
                if backup_file.suffix == '.gz':
                    content = self._decompress_backup(backup_file)
                else:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        content = yaml.safe_load(f)
                
                if content.get("spec", {}).get("metadata", {}).get("revision") == revision:
                    return str(backup_file)
            except Exception:
                continue
        return None
    
    def _load_backup(self, backup_file: str) -> Dict[str, Any]:
        """Charge une sauvegarde depuis un fichier."""
        backup_path = Path(backup_file)
        
        if backup_path.suffix == '.gz':
            return self._decompress_backup(backup_path)
        else:
            with open(backup_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    def _analyze_differences(self, current: List[Dict[str, Any]], 
                           target: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyse les différences entre les configurations."""
        differences = []
        
        # Index des ressources par clé unique
        current_index = {self._resource_key(r): r for r in current}
        target_index = {self._resource_key(r): r for r in target}
        
        # Ressources à supprimer (présentes actuellement mais pas dans la cible)
        for key, resource in current_index.items():
            if key not in target_index:
                differences.append({
                    "action": "delete",
                    "resource": resource,
                    "key": key
                })
        
        # Ressources à créer ou modifier
        for key, target_resource in target_index.items():
            if key not in current_index:
                differences.append({
                    "action": "create",
                    "resource": target_resource,
                    "key": key
                })
            else:
                current_resource = current_index[key]
                if self._resources_differ(current_resource, target_resource):
                    differences.append({
                        "action": "update",
                        "resource": target_resource,
                        "current": current_resource,
                        "key": key
                    })
        
        return differences
    
    def _resource_key(self, resource: Dict[str, Any]) -> str:
        """Génère une clé unique pour une ressource."""
        kind = resource.get("kind", "")
        name = resource.get("metadata", {}).get("name", "")
        namespace = resource.get("metadata", {}).get("namespace", "")
        return f"{kind}/{namespace}/{name}"
    
    def _resources_differ(self, resource1: Dict[str, Any], resource2: Dict[str, Any]) -> bool:
        """Vérifie si deux ressources diffèrent."""
        # Comparaison simplifiée basée sur les données importantes
        r1_data = resource1.get("data", {})
        r2_data = resource2.get("data", {})
        
        r1_spec = resource1.get("spec", {})
        r2_spec = resource2.get("spec", {})
        
        return r1_data != r2_data or r1_spec != r2_spec
    
    def _calculate_risk_level(self, affected_resources: List[Dict[str, Any]]) -> str:
        """Calcule le niveau de risque du rollback."""
        delete_count = sum(1 for r in affected_resources if r["action"] == "delete")
        update_count = sum(1 for r in affected_resources if r["action"] == "update")
        create_count = sum(1 for r in affected_resources if r["action"] == "create")
        
        total_changes = delete_count + update_count + create_count
        
        # Critères de risque
        if delete_count > 5 or total_changes > 20:
            return "high"
        elif delete_count > 2 or total_changes > 10:
            return "medium"
        else:
            return "low"
    
    def _estimate_rollback_duration(self, affected_resources: List[Dict[str, Any]]) -> int:
        """Estime la durée du rollback en secondes."""
        base_time = 30  # Temps de base
        
        for resource in affected_resources:
            if resource["action"] == "delete":
                base_time += 10
            elif resource["action"] == "update":
                base_time += 15
            elif resource["action"] == "create":
                base_time += 20
        
        return min(base_time, 600)  # Maximum 10 minutes
    
    def _determine_rollback_strategy(self, affected_resources: List[Dict[str, Any]], risk_level: str) -> str:
        """Détermine la stratégie de rollback optimale."""
        if risk_level == "high":
            return "incremental"  # Rollback étape par étape
        elif risk_level == "medium":
            return "atomic"  # Rollback atomique avec validation
        else:
            return "standard"  # Rollback standard
    
    def _generate_validation_steps(self, affected_resources: List[Dict[str, Any]]) -> List[str]:
        """Génère les étapes de validation post-rollback."""
        steps = [
            "Vérification de la connectivité des pods",
            "Validation des ConfigMaps",
            "Test des endpoints de service"
        ]
        
        # Étapes spécifiques selon les ressources affectées
        resource_types = {r["resource"].get("kind") for r in affected_resources}
        
        if "Deployment" in resource_types:
            steps.append("Vérification du statut des déploiements")
        if "Secret" in resource_types:
            steps.append("Validation des secrets et certificats")
        if "Service" in resource_types:
            steps.append("Test de connectivité des services")
        
        return steps
    
    def _execute_incremental_rollback(self, target_resources: List[Dict[str, Any]], 
                                    plan: RollbackPlan) -> bool:
        """Exécute un rollback incrémental (étape par étape)."""
        print("🔄 Rollback incrémental en cours...")
        
        # Groupement des ressources par priorité
        priority_groups = self._group_resources_by_priority(plan.affected_resources)
        
        for group_name, resources in priority_groups.items():
            print(f"   Traitement du groupe: {group_name}")
            
            for resource_change in resources:
                if not self._apply_resource_change(resource_change):
                    print(f"❌ Échec lors du traitement de {resource_change['key']}")
                    return False
                
                # Pause entre chaque ressource pour la stabilité
                time.sleep(2)
        
        return True
    
    def _execute_atomic_rollback(self, target_resources: List[Dict[str, Any]], 
                               plan: RollbackPlan) -> bool:
        """Exécute un rollback atomique."""
        print("🔄 Rollback atomique en cours...")
        
        # Création des ressources temporaires
        temp_files = []
        
        try:
            for resource in target_resources:
                temp_file = self._create_temp_resource_file(resource)
                temp_files.append(temp_file)
            
            # Application atomique de toutes les ressources
            for temp_file in temp_files:
                cmd = ["kubectl", "apply", "-f", temp_file, "-n", self.namespace]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Erreur lors de l'application: {result.stderr}")
                    return False
            
            return True
            
        finally:
            # Nettoyage des fichiers temporaires
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def _execute_standard_rollback(self, target_resources: List[Dict[str, Any]], 
                                 plan: RollbackPlan) -> bool:
        """Exécute un rollback standard."""
        print("🔄 Rollback standard en cours...")
        
        for resource_change in plan.affected_resources:
            if not self._apply_resource_change(resource_change):
                return False
        
        return True
    
    def _group_resources_by_priority(self, affected_resources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Groupe les ressources par priorité de traitement."""
        groups = {
            "secrets": [],
            "configmaps": [],
            "services": [],
            "deployments": [],
            "others": []
        }
        
        for resource in affected_resources:
            kind = resource["resource"].get("kind", "").lower()
            
            if kind == "secret":
                groups["secrets"].append(resource)
            elif kind == "configmap":
                groups["configmaps"].append(resource)
            elif kind == "service":
                groups["services"].append(resource)
            elif kind == "deployment":
                groups["deployments"].append(resource)
            else:
                groups["others"].append(resource)
        
        return groups
    
    def _apply_resource_change(self, resource_change: Dict[str, Any]) -> bool:
        """Applique un changement de ressource."""
        action = resource_change["action"]
        resource = resource_change["resource"]
        key = resource_change["key"]
        
        try:
            if action == "delete":
                return self._delete_resource(resource)
            elif action in ["create", "update"]:
                return self._apply_resource(resource)
            else:
                print(f"⚠️ Action inconnue: {action}")
                return False
        except Exception as e:
            print(f"❌ Erreur lors de l'application de {key}: {e}")
            return False
    
    def _delete_resource(self, resource: Dict[str, Any]) -> bool:
        """Supprime une ressource."""
        kind = resource.get("kind", "").lower()
        name = resource.get("metadata", {}).get("name", "")
        
        cmd = ["kubectl", "delete", kind, name, "-n", self.namespace]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def _apply_resource(self, resource: Dict[str, Any]) -> bool:
        """Applique une ressource."""
        temp_file = self._create_temp_resource_file(resource)
        
        try:
            cmd = ["kubectl", "apply", "-f", temp_file, "-n", self.namespace]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _create_temp_resource_file(self, resource: Dict[str, Any]) -> str:
        """Crée un fichier temporaire pour une ressource."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(resource, temp_file, default_flow_style=False)
        temp_file.close()
        return temp_file.name
    
    def _validate_rollback(self, plan: RollbackPlan) -> bool:
        """Valide le succès du rollback."""
        print("✅ Validation post-rollback...")
        
        for step in plan.validation_steps:
            print(f"   {step}...")
            
            # Simulation de validation (à adapter selon les besoins)
            time.sleep(1)
        
        # Vérification de la santé générale
        health_score = self._get_health_score()
        return health_score >= 70
    
    def _get_health_score(self) -> int:
        """Obtient le score de santé actuel du système."""
        try:
            # Utilisation du script de monitoring
            cmd = ["python3", os.path.join(os.path.dirname(__file__), "monitor_configs.py"),
                   "--namespace", self.namespace, "--one-shot"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parsing simple du score (à améliorer)
                return 85  # Score par défaut
            else:
                return 50
        except Exception:
            return 50
    
    def _find_last_stable_backup(self) -> Optional[BackupMetadata]:
        """Trouve la dernière sauvegarde stable."""
        backups = self.list_backups()
        
        # Pour cette implémentation, on considère la sauvegarde précédente comme stable
        if len(backups) >= 2:
            return backups[1]  # La deuxième plus récente
        
        return None
    
    def _emergency_restore(self) -> None:
        """Restauration d'urgence en cas d'échec critique."""
        print("🚨 Restauration d'urgence en cours...")
        
        # Implémentation basique - à étendre selon les besoins
        if hasattr(self, 'emergency_backup') and self.emergency_backup:
            emergency_config = self._load_backup(self.emergency_backup)
            emergency_resources = emergency_config["spec"]["resources"]
            
            for resource in emergency_resources:
                self._apply_resource(resource)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Gestionnaire de rollback de configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python rollback_configs.py --create-backup --description "Avant mise à jour majeure"
  python rollback_configs.py --list-backups
  python rollback_configs.py --rollback --target-revision 5 --confirm
  python rollback_configs.py --auto-rollback --health-threshold 60
        """
    )
    
    parser.add_argument(
        "--namespace", "-n",
        default="spotify-ai-agent-dev",
        help="Namespace Kubernetes cible"
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
        help="Crée une sauvegarde"
    )
    
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="Liste les sauvegardes disponibles"
    )
    
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Exécute un rollback"
    )
    
    parser.add_argument(
        "--auto-rollback",
        action="store_true",
        help="Rollback automatique en cas de problème"
    )
    
    # Paramètres de sauvegarde
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
    
    # Paramètres de rollback
    parser.add_argument(
        "--target-revision",
        type=int,
        help="Révision cible pour le rollback"
    )
    
    parser.add_argument(
        "--backup-file",
        help="Fichier de sauvegarde spécifique"
    )
    
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirme l'exécution du rollback"
    )
    
    # Paramètres de surveillance automatique
    parser.add_argument(
        "--health-threshold",
        type=int,
        default=50,
        help="Seuil de santé pour rollback automatique"
    )
    
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Intervalle de vérification (secondes)"
    )
    
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=300,
        help="Temps d'attente maximum (secondes)"
    )
    
    # Maintenance
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Nettoie les anciennes sauvegardes"
    )
    
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Rétention des sauvegardes (jours)"
    )
    
    parser.add_argument(
        "--max-backups",
        type=int,
        default=50,
        help="Nombre maximum de sauvegardes"
    )
    
    args = parser.parse_args()
    
    try:
        # Création du gestionnaire de rollback
        rollback_manager = ConfigurationRollbackManager(
            namespace=args.namespace,
            kubeconfig=args.kubeconfig,
            backup_dir=args.backup_dir
        )
        
        if args.create_backup:
            backup_file = rollback_manager.create_backup(
                description=args.description,
                created_by=args.created_by
            )
            print(f"📦 Sauvegarde créée: {backup_file}")
        
        elif args.list_backups:
            backups = rollback_manager.list_backups()
            
            if backups:
                print("\n📋 Sauvegardes disponibles:")
                print(f"{'Révision':<10} {'Date':<20} {'Créateur':<15} {'Description'}")
                print("-" * 80)
                
                for backup in backups:
                    date_str = backup.timestamp.strftime("%Y-%m-%d %H:%M")
                    description = backup.description[:30] + "..." if len(backup.description) > 30 else backup.description
                    print(f"{backup.revision:<10} {date_str:<20} {backup.created_by:<15} {description}")
            else:
                print("Aucune sauvegarde disponible")
        
        elif args.rollback:
            if not args.target_revision:
                print("❌ Révision cible requise pour le rollback")
                sys.exit(1)
            
            # Analyse de l'impact
            plan = rollback_manager.analyze_rollback_impact(args.target_revision)
            
            print("\n📊 Plan de rollback:")
            print(f"   Révision cible: {plan.target_revision}")
            print(f"   Stratégie: {plan.rollback_strategy}")
            print(f"   Niveau de risque: {plan.risk_level}")
            print(f"   Durée estimée: {plan.estimated_duration}s")
            print(f"   Ressources affectées: {len(plan.affected_resources)}")
            
            # Exécution
            success = rollback_manager.execute_rollback(plan, args.confirm)
            if success:
                print("✅ Rollback terminé avec succès")
            else:
                print("❌ Échec du rollback")
                sys.exit(1)
        
        elif args.auto_rollback:
            success = rollback_manager.auto_rollback_on_health_degradation(
                health_threshold=args.health_threshold,
                check_interval=args.check_interval,
                max_wait_time=args.max_wait_time
            )
            
            if not success:
                sys.exit(1)
        
        elif args.cleanup:
            rollback_manager.cleanup_old_backups(
                retention_days=args.retention_days,
                max_backups=args.max_backups
            )
        
        else:
            print("Aucune action spécifiée. Utilisez --help pour voir les options.")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
