#!/usr/bin/env python3
"""
Configuration Deployment Script
==============================

Script pour déployer les configurations dans un cluster Kubernetes.
Gère le déploiement des ConfigMaps, Secrets et autres ressources.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python deploy_configs.py [options]
    
Examples:
    python deploy_configs.py --namespace spotify-ai-agent-dev
    python deploy_configs.py --config-dir ./configs/ --dry-run
    python deploy_configs.py --apply --wait-for-rollout
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class KubernetesDeploymentError(Exception):
    """Exception pour les erreurs de déploiement Kubernetes."""
    pass

class ConfigurationDeployer:
    """Déployeur de configurations Kubernetes."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 kubeconfig: Optional[str] = None,
                 dry_run: bool = False):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.dry_run = dry_run
        self.applied_resources = []
        self.failed_resources = []
    
    def check_prerequisites(self) -> bool:
        """Vérifie les prérequis pour le déploiement."""
        print("🔍 Vérification des prérequis...")
        
        # Vérification de kubectl
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ kubectl disponible")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ kubectl non trouvé ou non fonctionnel")
            return False
        
        # Vérification de l'accès au cluster
        try:
            cmd = ["kubectl", "cluster-info"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Connexion au cluster Kubernetes")
        except subprocess.CalledProcessError as e:
            print(f"❌ Impossible de se connecter au cluster: {e}")
            return False
        
        # Vérification/création du namespace
        if not self._ensure_namespace_exists():
            return False
        
        # Vérification des permissions
        if not self._check_permissions():
            return False
        
        return True
    
    def _ensure_namespace_exists(self) -> bool:
        """S'assure que le namespace existe."""
        try:
            cmd = ["kubectl", "get", "namespace", self.namespace]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ Namespace '{self.namespace}' existe")
            return True
        except subprocess.CalledProcessError:
            # Création du namespace
            if self.dry_run:
                print(f"🔍 [DRY-RUN] Créerait le namespace '{self.namespace}'")
                return True
            
            try:
                cmd = ["kubectl", "create", "namespace", self.namespace]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"✅ Namespace '{self.namespace}' créé")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Impossible de créer le namespace: {e}")
                return False
    
    def _check_permissions(self) -> bool:
        """Vérifie les permissions nécessaires."""
        resources_to_check = ["configmaps", "secrets", "deployments", "services"]
        
        for resource in resources_to_check:
            try:
                cmd = [
                    "kubectl", "auth", "can-i", "create", resource,
                    "--namespace", self.namespace
                ]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or "yes" not in result.stdout.lower():
                    print(f"❌ Permissions insuffisantes pour {resource}")
                    return False
            except subprocess.CalledProcessError:
                print(f"❌ Impossible de vérifier les permissions pour {resource}")
                return False
        
        print("✅ Permissions suffisantes")
        return True
    
    def load_configurations(self, config_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Charge toutes les configurations depuis un répertoire."""
        configurations = {
            "configmaps": [],
            "secrets": [],
            "other": []
        }
        
        # Recherche des fichiers YAML/JSON
        config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load_all(f)
                    
                    for doc in content:
                        if not doc:
                            continue
                        
                        # Classification par type de ressource
                        kind = doc.get("kind", "").lower()
                        if kind == "configmap":
                            configurations["configmaps"].append(doc)
                        elif kind == "secret":
                            configurations["secrets"].append(doc)
                        else:
                            configurations["other"].append(doc)
                            
            except Exception as e:
                print(f"⚠️ Erreur lors de la lecture de {config_file}: {e}")
        
        return configurations
    
    def validate_resources(self, configurations: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Valide les ressources avant déploiement."""
        print("📋 Validation des ressources...")
        
        all_valid = True
        total_resources = sum(len(resources) for resources in configurations.values())
        
        print(f"📊 Ressources à déployer:")
        for resource_type, resources in configurations.items():
            if resources:
                print(f"  • {resource_type}: {len(resources)}")
        
        # Validation des ConfigMaps
        for configmap in configurations["configmaps"]:
            if not self._validate_configmap(configmap):
                all_valid = False
        
        # Validation des Secrets
        for secret in configurations["secrets"]:
            if not self._validate_secret(secret):
                all_valid = False
        
        if all_valid:
            print(f"✅ Toutes les {total_resources} ressources sont valides")
        else:
            print("❌ Certaines ressources ne sont pas valides")
        
        return all_valid
    
    def _validate_configmap(self, configmap: Dict[str, Any]) -> bool:
        """Valide une ConfigMap."""
        required_fields = ["apiVersion", "kind", "metadata", "data"]
        
        for field in required_fields:
            if field not in configmap:
                print(f"❌ ConfigMap manque le champ requis: {field}")
                return False
        
        if configmap["kind"] != "ConfigMap":
            print(f"❌ Type de ressource incorrect: {configmap['kind']}")
            return False
        
        if "name" not in configmap["metadata"]:
            print("❌ ConfigMap manque metadata.name")
            return False
        
        # Validation du namespace
        if "namespace" in configmap["metadata"]:
            if configmap["metadata"]["namespace"] != self.namespace:
                print(f"⚠️ ConfigMap a un namespace différent: {configmap['metadata']['namespace']}")
        else:
            configmap["metadata"]["namespace"] = self.namespace
        
        return True
    
    def _validate_secret(self, secret: Dict[str, Any]) -> bool:
        """Valide un Secret."""
        required_fields = ["apiVersion", "kind", "metadata"]
        
        for field in required_fields:
            if field not in secret:
                print(f"❌ Secret manque le champ requis: {field}")
                return False
        
        if secret["kind"] != "Secret":
            print(f"❌ Type de ressource incorrect: {secret['kind']}")
            return False
        
        if "name" not in secret["metadata"]:
            print("❌ Secret manque metadata.name")
            return False
        
        # Validation du namespace
        if "namespace" in secret["metadata"]:
            if secret["metadata"]["namespace"] != self.namespace:
                print(f"⚠️ Secret a un namespace différent: {secret['metadata']['namespace']}")
        else:
            secret["metadata"]["namespace"] = self.namespace
        
        return True
    
    def deploy_configurations(self, configurations: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Déploie toutes les configurations."""
        print(f"🚀 Déploiement des configurations dans le namespace '{self.namespace}'...")
        
        if self.dry_run:
            print("🔍 Mode DRY-RUN activé - aucune ressource ne sera réellement créée")
        
        success = True
        
        # Déploiement des ConfigMaps en premier
        if configurations["configmaps"]:
            print("\n📋 Déploiement des ConfigMaps...")
            for configmap in configurations["configmaps"]:
                if not self._apply_resource(configmap):
                    success = False
        
        # Déploiement des Secrets
        if configurations["secrets"]:
            print("\n🔐 Déploiement des Secrets...")
            for secret in configurations["secrets"]:
                if not self._apply_resource(secret):
                    success = False
        
        # Déploiement des autres ressources
        if configurations["other"]:
            print("\n📦 Déploiement des autres ressources...")
            for resource in configurations["other"]:
                if not self._apply_resource(resource):
                    success = False
        
        return success
    
    def _apply_resource(self, resource: Dict[str, Any]) -> bool:
        """Applique une ressource Kubernetes."""
        resource_name = resource.get("metadata", {}).get("name", "unknown")
        resource_kind = resource.get("kind", "unknown")
        
        try:
            # Création du fichier temporaire
            temp_file = f"/tmp/{resource_kind.lower()}_{resource_name}.yaml"
            with open(temp_file, 'w', encoding='utf-8') as f:
                yaml.dump(resource, f, default_flow_style=False)
            
            # Construction de la commande kubectl
            cmd = ["kubectl", "apply", "-f", temp_file]
            if self.dry_run:
                cmd.append("--dry-run=client")
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            # Exécution de la commande
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            action = "validé" if self.dry_run else "appliqué"
            print(f"  ✅ {resource_kind} '{resource_name}' {action}")
            
            self.applied_resources.append({
                "kind": resource_kind,
                "name": resource_name,
                "namespace": self.namespace
            })
            
            # Nettoyage du fichier temporaire
            os.unlink(temp_file)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Erreur lors de l'application de {resource_kind} '{resource_name}': {e.stderr}")
            self.failed_resources.append({
                "kind": resource_kind,
                "name": resource_name,
                "error": str(e.stderr)
            })
            return False
        except Exception as e:
            print(f"  ❌ Erreur générale pour {resource_kind} '{resource_name}': {e}")
            self.failed_resources.append({
                "kind": resource_kind,
                "name": resource_name,
                "error": str(e)
            })
            return False
    
    def wait_for_rollout(self, timeout: int = 300) -> bool:
        """Attend la fin du déploiement des ressources."""
        if self.dry_run:
            print("🔍 [DRY-RUN] Simulation de l'attente du rollout")
            return True
        
        print(f"⏳ Attente du déploiement complet (timeout: {timeout}s)...")
        
        deployments = self._get_deployments()
        if not deployments:
            print("ℹ️ Aucun déploiement à surveiller")
            return True
        
        start_time = time.time()
        
        for deployment in deployments:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                print(f"❌ Timeout atteint lors de l'attente du déploiement")
                return False
            
            if not self._wait_for_deployment_rollout(deployment, int(remaining_time)):
                return False
        
        print("✅ Tous les déploiements sont terminés avec succès")
        return True
    
    def _get_deployments(self) -> List[str]:
        """Récupère la liste des déploiements dans le namespace."""
        try:
            cmd = ["kubectl", "get", "deployments", "-n", self.namespace, "-o", "name"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployments = [line.strip().replace("deployment.apps/", "") 
                          for line in result.stdout.strip().split("\n") if line.strip()]
            return deployments
        except subprocess.CalledProcessError:
            return []
    
    def _wait_for_deployment_rollout(self, deployment: str, timeout: int) -> bool:
        """Attend le rollout d'un déploiement spécifique."""
        try:
            cmd = [
                "kubectl", "rollout", "status", f"deployment/{deployment}",
                "-n", self.namespace, f"--timeout={timeout}s"
            ]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ Déploiement '{deployment}' terminé")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Échec du déploiement '{deployment}': {e.stderr}")
            return False
    
    def verify_deployment(self) -> Dict[str, Any]:
        """Vérifie l'état du déploiement."""
        print("🔍 Vérification de l'état du déploiement...")
        
        verification_results = {
            "configmaps": {"total": 0, "ready": 0, "failed": 0},
            "secrets": {"total": 0, "ready": 0, "failed": 0},
            "deployments": {"total": 0, "ready": 0, "failed": 0},
            "overall_status": "unknown"
        }
        
        # Vérification des ConfigMaps
        configmaps = self._get_resources("configmaps")
        verification_results["configmaps"]["total"] = len(configmaps)
        verification_results["configmaps"]["ready"] = len(configmaps)  # ConfigMaps sont toujours prêtes si créées
        
        # Vérification des Secrets
        secrets = self._get_resources("secrets")
        verification_results["secrets"]["total"] = len(secrets)
        verification_results["secrets"]["ready"] = len(secrets)  # Secrets sont toujours prêts si créés
        
        # Vérification des Déploiements
        deployments = self._get_deployments()
        verification_results["deployments"]["total"] = len(deployments)
        
        for deployment in deployments:
            if self._is_deployment_ready(deployment):
                verification_results["deployments"]["ready"] += 1
            else:
                verification_results["deployments"]["failed"] += 1
        
        # Détermination du statut global
        total_resources = (verification_results["configmaps"]["total"] + 
                          verification_results["secrets"]["total"] + 
                          verification_results["deployments"]["total"])
        
        ready_resources = (verification_results["configmaps"]["ready"] + 
                          verification_results["secrets"]["ready"] + 
                          verification_results["deployments"]["ready"])
        
        if ready_resources == total_resources:
            verification_results["overall_status"] = "success"
            print("✅ Tous les composants sont déployés et fonctionnels")
        elif ready_resources > 0:
            verification_results["overall_status"] = "partial"
            print("⚠️ Déploiement partiellement réussi")
        else:
            verification_results["overall_status"] = "failed"
            print("❌ Échec du déploiement")
        
        return verification_results
    
    def _get_resources(self, resource_type: str) -> List[str]:
        """Récupère la liste des ressources d'un type donné."""
        try:
            cmd = ["kubectl", "get", resource_type, "-n", self.namespace, "-o", "name"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            resources = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            return resources
        except subprocess.CalledProcessError:
            return []
    
    def _is_deployment_ready(self, deployment: str) -> bool:
        """Vérifie si un déploiement est prêt."""
        try:
            cmd = [
                "kubectl", "get", f"deployment/{deployment}",
                "-n", self.namespace, "-o", "jsonpath={.status.readyReplicas}"
            ]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            ready_replicas = int(result.stdout.strip() or "0")
            
            # Récupération du nombre de répliques désirées
            cmd[6] = "jsonpath={.spec.replicas}"
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            desired_replicas = int(result.stdout.strip() or "0")
            
            return ready_replicas == desired_replicas and ready_replicas > 0
        except (subprocess.CalledProcessError, ValueError):
            return False
    
    def generate_deployment_summary(self) -> str:
        """Génère un résumé du déploiement."""
        summary = []
        summary.append("=" * 60)
        summary.append("RÉSUMÉ DU DÉPLOIEMENT")
        summary.append("=" * 60)
        summary.append(f"Timestamp: {datetime.now().isoformat()}")
        summary.append(f"Namespace: {self.namespace}")
        summary.append(f"Mode: {'DRY-RUN' if self.dry_run else 'APPLY'}")
        summary.append("")
        
        summary.append(f"Ressources appliquées avec succès: {len(self.applied_resources)}")
        for resource in self.applied_resources:
            summary.append(f"  ✅ {resource['kind']}/{resource['name']}")
        
        if self.failed_resources:
            summary.append(f"\nRessources échouées: {len(self.failed_resources)}")
            for resource in self.failed_resources:
                summary.append(f"  ❌ {resource['kind']}/{resource['name']}: {resource['error']}")
        
        summary.append("")
        summary.append("=" * 60)
        
        return "\n".join(summary)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Déployeur de configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python deploy_configs.py --namespace spotify-ai-agent-dev
  python deploy_configs.py --config-dir ./configs/ --dry-run
  python deploy_configs.py --apply --wait-for-rollout --timeout 600
        """
    )
    
    parser.add_argument(
        "--config-dir", "-d",
        type=Path,
        default=Path("./configs"),
        help="Répertoire contenant les configurations à déployer"
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
        "--dry-run",
        action="store_true",
        help="Mode simulation - ne déploie pas réellement"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Applique réellement les configurations (requis pour le déploiement réel)"
    )
    
    parser.add_argument(
        "--wait-for-rollout",
        action="store_true",
        help="Attend la fin du déploiement"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout pour l'attente du rollout (secondes, défaut: 300)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifie l'état du déploiement après application"
    )
    
    parser.add_argument(
        "--output-summary",
        type=Path,
        help="Fichier de sortie pour le résumé du déploiement"
    )
    
    args = parser.parse_args()
    
    # Validation des arguments
    if not args.dry_run and not args.apply:
        print("❌ Erreur: Utilisez --apply pour un déploiement réel ou --dry-run pour une simulation")
        sys.exit(1)
    
    if not args.config_dir.exists():
        print(f"❌ Erreur: Le répertoire {args.config_dir} n'existe pas")
        sys.exit(1)
    
    try:
        # Création du déployeur
        deployer = ConfigurationDeployer(
            namespace=args.namespace,
            kubeconfig=args.kubeconfig,
            dry_run=args.dry_run
        )
        
        # Vérification des prérequis
        if not deployer.check_prerequisites():
            print("❌ Prérequis non satisfaits")
            sys.exit(1)
        
        # Chargement des configurations
        configurations = deployer.load_configurations(args.config_dir)
        
        if not any(configurations.values()):
            print("⚠️ Aucune configuration trouvée à déployer")
            sys.exit(0)
        
        # Validation des ressources
        if not deployer.validate_resources(configurations):
            print("❌ Validation des ressources échouée")
            sys.exit(1)
        
        # Déploiement
        if not deployer.deploy_configurations(configurations):
            print("❌ Déploiement échoué")
            sys.exit(1)
        
        # Attente du rollout
        if args.wait_for_rollout and not args.dry_run:
            if not deployer.wait_for_rollout(args.timeout):
                print("❌ Timeout lors de l'attente du rollout")
                sys.exit(1)
        
        # Vérification
        if args.verify and not args.dry_run:
            verification_results = deployer.verify_deployment()
            if verification_results["overall_status"] != "success":
                print("⚠️ Vérification: des problèmes ont été détectés")
        
        # Génération du résumé
        summary = deployer.generate_deployment_summary()
        print("\n" + summary)
        
        # Sauvegarde du résumé
        if args.output_summary:
            with open(args.output_summary, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\n📄 Résumé sauvegardé: {args.output_summary}")
        
        print("\n🎉 Déploiement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du déploiement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
