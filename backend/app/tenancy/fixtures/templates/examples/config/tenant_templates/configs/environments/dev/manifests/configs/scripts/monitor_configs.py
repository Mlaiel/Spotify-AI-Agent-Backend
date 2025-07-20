#!/usr/bin/env python3
"""
Configuration Monitoring Script
==============================

Script pour surveiller l'état des configurations déployées.
Surveille les ConfigMaps, Secrets et déploiements en temps réel.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python monitor_configs.py [options]
    
Examples:
    python monitor_configs.py --namespace spotify-ai-agent-dev
    python monitor_configs.py --watch --interval 30
    python monitor_configs.py --export-metrics --prometheus-format
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import threading

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfigurationMonitor:
    """Moniteur de configurations Kubernetes."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 kubeconfig: Optional[str] = None,
                 watch_interval: int = 30):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.watch_interval = watch_interval
        self.monitoring = False
        self.metrics_history = defaultdict(list)
        self.last_check = None
        self.alerts = []
    
    def check_cluster_connectivity(self) -> bool:
        """Vérifie la connectivité au cluster."""
        try:
            cmd = ["kubectl", "cluster-info"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Récupère l'état complet des configurations."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "namespace": self.namespace,
            "configmaps": self._get_configmaps_status(),
            "secrets": self._get_secrets_status(),
            "deployments": self._get_deployments_status(),
            "pods": self._get_pods_status(),
            "services": self._get_services_status(),
            "health_score": 0,
            "alerts": []
        }
        
        # Calcul du score de santé
        status["health_score"] = self._calculate_health_score(status)
        
        # Détection d'alertes
        status["alerts"] = self._detect_alerts(status)
        
        return status
    
    def _get_configmaps_status(self) -> Dict[str, Any]:
        """Récupère l'état des ConfigMaps."""
        try:
            cmd = ["kubectl", "get", "configmaps", "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            configmaps = []
            for item in data.get("items", []):
                configmap_info = {
                    "name": item.get("metadata", {}).get("name"),
                    "age": self._calculate_age(item.get("metadata", {}).get("creationTimestamp")),
                    "data_keys": len(item.get("data", {})),
                    "size_bytes": self._calculate_configmap_size(item.get("data", {})),
                    "labels": item.get("metadata", {}).get("labels", {}),
                    "annotations": item.get("metadata", {}).get("annotations", {})
                }
                configmaps.append(configmap_info)
            
            return {
                "total": len(configmaps),
                "details": configmaps,
                "status": "healthy" if configmaps else "warning"
            }
        except Exception as e:
            return {
                "total": 0,
                "details": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_secrets_status(self) -> Dict[str, Any]:
        """Récupère l'état des Secrets."""
        try:
            cmd = ["kubectl", "get", "secrets", "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            secrets = []
            for item in data.get("items", []):
                # Filtrer les secrets système
                if item.get("type") == "kubernetes.io/service-account-token":
                    continue
                
                secret_info = {
                    "name": item.get("metadata", {}).get("name"),
                    "type": item.get("type"),
                    "age": self._calculate_age(item.get("metadata", {}).get("creationTimestamp")),
                    "data_keys": len(item.get("data", {})),
                    "labels": item.get("metadata", {}).get("labels", {}),
                    "annotations": item.get("metadata", {}).get("annotations", {})
                }
                secrets.append(secret_info)
            
            return {
                "total": len(secrets),
                "details": secrets,
                "status": "healthy" if secrets else "warning"
            }
        except Exception as e:
            return {
                "total": 0,
                "details": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_deployments_status(self) -> Dict[str, Any]:
        """Récupère l'état des Déploiements."""
        try:
            cmd = ["kubectl", "get", "deployments", "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            deployments = []
            for item in data.get("items", []):
                status = item.get("status", {})
                spec = item.get("spec", {})
                
                deployment_info = {
                    "name": item.get("metadata", {}).get("name"),
                    "age": self._calculate_age(item.get("metadata", {}).get("creationTimestamp")),
                    "replicas": {
                        "desired": spec.get("replicas", 0),
                        "current": status.get("replicas", 0),
                        "ready": status.get("readyReplicas", 0),
                        "available": status.get("availableReplicas", 0),
                        "unavailable": status.get("unavailableReplicas", 0)
                    },
                    "conditions": status.get("conditions", []),
                    "strategy": spec.get("strategy", {}),
                    "selector": spec.get("selector", {}),
                    "status": self._determine_deployment_status(status, spec)
                }
                deployments.append(deployment_info)
            
            # Calcul du statut global
            if not deployments:
                overall_status = "warning"
            elif all(d["status"] == "ready" for d in deployments):
                overall_status = "healthy"
            elif any(d["status"] == "failed" for d in deployments):
                overall_status = "error"
            else:
                overall_status = "warning"
            
            return {
                "total": len(deployments),
                "details": deployments,
                "status": overall_status
            }
        except Exception as e:
            return {
                "total": 0,
                "details": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_pods_status(self) -> Dict[str, Any]:
        """Récupère l'état des Pods."""
        try:
            cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            pods = []
            phase_counts = defaultdict(int)
            
            for item in data.get("items", []):
                status = item.get("status", {})
                spec = item.get("spec", {})
                
                pod_phase = status.get("phase", "Unknown")
                phase_counts[pod_phase] += 1
                
                # Calcul des restarts
                restart_count = sum(
                    container.get("restartCount", 0)
                    for container in status.get("containerStatuses", [])
                )
                
                pod_info = {
                    "name": item.get("metadata", {}).get("name"),
                    "phase": pod_phase,
                    "age": self._calculate_age(item.get("metadata", {}).get("creationTimestamp")),
                    "restart_count": restart_count,
                    "node": spec.get("nodeName"),
                    "containers": len(spec.get("containers", [])),
                    "ready_containers": len([
                        c for c in status.get("containerStatuses", [])
                        if c.get("ready", False)
                    ]),
                    "conditions": status.get("conditions", [])
                }
                pods.append(pod_info)
            
            return {
                "total": len(pods),
                "phase_counts": dict(phase_counts),
                "details": pods,
                "status": "healthy" if phase_counts.get("Running", 0) == len(pods) else "warning"
            }
        except Exception as e:
            return {
                "total": 0,
                "phase_counts": {},
                "details": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_services_status(self) -> Dict[str, Any]:
        """Récupère l'état des Services."""
        try:
            cmd = ["kubectl", "get", "services", "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            services = []
            for item in data.get("items", []):
                spec = item.get("spec", {})
                status = item.get("status", {})
                
                service_info = {
                    "name": item.get("metadata", {}).get("name"),
                    "type": spec.get("type", "ClusterIP"),
                    "age": self._calculate_age(item.get("metadata", {}).get("creationTimestamp")),
                    "cluster_ip": spec.get("clusterIP"),
                    "external_ip": spec.get("externalIPs", []),
                    "ports": spec.get("ports", []),
                    "selector": spec.get("selector", {}),
                    "endpoints": self._get_service_endpoints(item.get("metadata", {}).get("name"))
                }
                services.append(service_info)
            
            return {
                "total": len(services),
                "details": services,
                "status": "healthy" if services else "warning"
            }
        except Exception as e:
            return {
                "total": 0,
                "details": [],
                "status": "error",
                "error": str(e)
            }
    
    def _get_service_endpoints(self, service_name: str) -> List[str]:
        """Récupère les endpoints d'un service."""
        try:
            cmd = ["kubectl", "get", "endpoints", service_name, "-n", self.namespace, "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            endpoints = []
            for subset in data.get("subsets", []):
                for address in subset.get("addresses", []):
                    for port in subset.get("ports", []):
                        endpoints.append(f"{address.get('ip')}:{port.get('port')}")
            
            return endpoints
        except Exception:
            return []
    
    def _calculate_age(self, creation_timestamp: str) -> str:
        """Calcule l'âge d'une ressource."""
        if not creation_timestamp:
            return "unknown"
        
        try:
            # Parse ISO format
            created = datetime.fromisoformat(creation_timestamp.replace('Z', '+00:00'))
            now = datetime.now(created.tzinfo)
            age = now - created
            
            if age.days > 0:
                return f"{age.days}d"
            elif age.seconds > 3600:
                hours = age.seconds // 3600
                return f"{hours}h"
            elif age.seconds > 60:
                minutes = age.seconds // 60
                return f"{minutes}m"
            else:
                return f"{age.seconds}s"
        except Exception:
            return "unknown"
    
    def _calculate_configmap_size(self, data: Dict[str, str]) -> int:
        """Calcule la taille d'une ConfigMap en bytes."""
        total_size = 0
        for key, value in data.items():
            total_size += len(key.encode('utf-8')) + len(value.encode('utf-8'))
        return total_size
    
    def _determine_deployment_status(self, status: Dict, spec: Dict) -> str:
        """Détermine le statut d'un déploiement."""
        desired = spec.get("replicas", 0)
        ready = status.get("readyReplicas", 0)
        available = status.get("availableReplicas", 0)
        
        if ready == desired and available == desired and desired > 0:
            return "ready"
        elif ready == 0:
            return "failed"
        elif ready < desired:
            return "scaling"
        else:
            return "unknown"
    
    def _calculate_health_score(self, status: Dict[str, Any]) -> float:
        """Calcule un score de santé global (0-100)."""
        score = 100.0
        
        # Pénalités pour les erreurs
        for component in ["configmaps", "secrets", "deployments", "pods", "services"]:
            if status[component]["status"] == "error":
                score -= 20
            elif status[component]["status"] == "warning":
                score -= 10
        
        # Pénalités pour les déploiements non prêts
        for deployment in status["deployments"].get("details", []):
            if deployment["status"] != "ready":
                score -= 15
        
        # Pénalités pour les pods non running
        pod_phase_counts = status["pods"].get("phase_counts", {})
        total_pods = status["pods"]["total"]
        if total_pods > 0:
            running_ratio = pod_phase_counts.get("Running", 0) / total_pods
            score -= (1 - running_ratio) * 20
        
        return max(0, score)
    
    def _detect_alerts(self, status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Détecte les alertes basées sur l'état."""
        alerts = []
        
        # Alertes pour les déploiements
        for deployment in status["deployments"].get("details", []):
            if deployment["status"] == "failed":
                alerts.append({
                    "severity": "critical",
                    "type": "deployment_failed",
                    "resource": deployment["name"],
                    "message": f"Déploiement '{deployment['name']}' en échec"
                })
            elif deployment["replicas"]["ready"] < deployment["replicas"]["desired"]:
                alerts.append({
                    "severity": "warning",
                    "type": "deployment_scaling",
                    "resource": deployment["name"],
                    "message": f"Déploiement '{deployment['name']}' en cours de mise à l'échelle"
                })
        
        # Alertes pour les pods
        for pod in status["pods"].get("details", []):
            if pod["restart_count"] > 5:
                alerts.append({
                    "severity": "warning",
                    "type": "high_restart_count",
                    "resource": pod["name"],
                    "message": f"Pod '{pod['name']}' a redémarré {pod['restart_count']} fois"
                })
            elif pod["phase"] not in ["Running", "Succeeded"]:
                alerts.append({
                    "severity": "critical",
                    "type": "pod_not_running",
                    "resource": pod["name"],
                    "message": f"Pod '{pod['name']}' en phase {pod['phase']}"
                })
        
        # Alerte pour le score de santé faible
        if status["health_score"] < 50:
            alerts.append({
                "severity": "critical",
                "type": "low_health_score",
                "resource": "cluster",
                "message": f"Score de santé faible: {status['health_score']:.1f}%"
            })
        elif status["health_score"] < 80:
            alerts.append({
                "severity": "warning",
                "type": "degraded_health",
                "resource": "cluster",
                "message": f"Santé dégradée: {status['health_score']:.1f}%"
            })
        
        return alerts
    
    def start_monitoring(self, duration: Optional[int] = None) -> None:
        """Démarre la surveillance en continu."""
        print(f"🔍 Démarrage de la surveillance du namespace '{self.namespace}'")
        print(f"⏱️ Intervalle de vérification: {self.watch_interval}s")
        
        if duration:
            print(f"⏰ Durée de surveillance: {duration}s")
        
        self.monitoring = True
        start_time = time.time()
        
        # Gestionnaire de signal pour arrêt propre
        def signal_handler(signum, frame):
            print("\n🛑 Arrêt de la surveillance...")
            self.monitoring = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.monitoring:
                # Vérification de la durée
                if duration and (time.time() - start_time) >= duration:
                    print(f"⏰ Durée de surveillance atteinte ({duration}s)")
                    break
                
                # Récupération du statut
                status = self.get_configuration_status()
                self.last_check = status
                
                # Affichage du statut
                self._display_status_summary(status)
                
                # Sauvegarde de l'historique
                self._save_metrics_history(status)
                
                # Affichage des alertes
                if status["alerts"]:
                    self._display_alerts(status["alerts"])
                
                # Attente avant la prochaine vérification
                if self.monitoring:
                    time.sleep(self.watch_interval)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.monitoring = False
            print("🏁 Surveillance terminée")
    
    def _display_status_summary(self, status: Dict[str, Any]) -> None:
        """Affiche un résumé du statut."""
        timestamp = status["timestamp"]
        health_score = status["health_score"]
        
        # Icône de santé
        if health_score >= 90:
            health_icon = "💚"
        elif health_score >= 70:
            health_icon = "💛"
        else:
            health_icon = "❤️"
        
        print(f"\n{health_icon} [{timestamp}] Score de santé: {health_score:.1f}%")
        
        # Résumé des composants
        components = ["configmaps", "secrets", "deployments", "pods", "services"]
        for component in components:
            total = status[component]["total"]
            comp_status = status[component]["status"]
            
            if comp_status == "healthy":
                icon = "✅"
            elif comp_status == "warning":
                icon = "⚠️"
            else:
                icon = "❌"
            
            print(f"  {icon} {component.capitalize()}: {total}")
        
        # Détails des déploiements
        for deployment in status["deployments"].get("details", []):
            ready = deployment["replicas"]["ready"]
            desired = deployment["replicas"]["desired"]
            name = deployment["name"]
            
            if deployment["status"] == "ready":
                icon = "🟢"
            elif deployment["status"] == "scaling":
                icon = "🟡"
            else:
                icon = "🔴"
            
            print(f"    {icon} {name}: {ready}/{desired}")
    
    def _display_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Affiche les alertes."""
        if not alerts:
            return
        
        print(f"\n🚨 Alertes ({len(alerts)}):")
        for alert in alerts:
            severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
            print(f"  {severity_icon} {alert['message']}")
    
    def _save_metrics_history(self, status: Dict[str, Any]) -> None:
        """Sauvegarde l'historique des métriques."""
        timestamp = datetime.now()
        
        # Limitation de l'historique à 24h
        cutoff = timestamp - timedelta(hours=24)
        
        # Nettoyage de l'ancien historique
        for metric_name in self.metrics_history:
            self.metrics_history[metric_name] = [
                entry for entry in self.metrics_history[metric_name]
                if entry["timestamp"] > cutoff
            ]
        
        # Ajout des nouvelles métriques
        self.metrics_history["health_score"].append({
            "timestamp": timestamp,
            "value": status["health_score"]
        })
        
        self.metrics_history["total_pods"].append({
            "timestamp": timestamp,
            "value": status["pods"]["total"]
        })
        
        self.metrics_history["running_pods"].append({
            "timestamp": timestamp,
            "value": status["pods"]["phase_counts"].get("Running", 0)
        })
        
        self.metrics_history["total_deployments"].append({
            "timestamp": timestamp,
            "value": status["deployments"]["total"]
        })
        
        ready_deployments = sum(
            1 for d in status["deployments"].get("details", [])
            if d["status"] == "ready"
        )
        self.metrics_history["ready_deployments"].append({
            "timestamp": timestamp,
            "value": ready_deployments
        })
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Exporte les métriques dans différents formats."""
        if format_type == "prometheus":
            return self._export_prometheus_metrics()
        elif format_type == "json":
            return self._export_json_metrics()
        elif format_type == "csv":
            return self._export_csv_metrics()
        else:
            raise ValueError(f"Format non supporté: {format_type}")
    
    def _export_prometheus_metrics(self) -> str:
        """Exporte les métriques au format Prometheus."""
        lines = []
        
        if self.last_check:
            status = self.last_check
            
            # Métriques de santé
            lines.append(f"# HELP spotify_ai_agent_health_score Score de santé général du cluster")
            lines.append(f"# TYPE spotify_ai_agent_health_score gauge")
            lines.append(f'spotify_ai_agent_health_score{{namespace="{self.namespace}"}} {status["health_score"]}')
            
            # Métriques des pods
            for phase, count in status["pods"]["phase_counts"].items():
                lines.append(f'spotify_ai_agent_pods_total{{namespace="{self.namespace}",phase="{phase}"}} {count}')
            
            # Métriques des déploiements
            for deployment in status["deployments"].get("details", []):
                name = deployment["name"]
                desired = deployment["replicas"]["desired"]
                ready = deployment["replicas"]["ready"]
                
                lines.append(f'spotify_ai_agent_deployment_replicas_desired{{namespace="{self.namespace}",deployment="{name}"}} {desired}')
                lines.append(f'spotify_ai_agent_deployment_replicas_ready{{namespace="{self.namespace}",deployment="{name}"}} {ready}')
        
        return "\n".join(lines)
    
    def _export_json_metrics(self) -> str:
        """Exporte les métriques au format JSON."""
        return json.dumps({
            "current_status": self.last_check,
            "metrics_history": {
                k: [{"timestamp": entry["timestamp"].isoformat(), "value": entry["value"]}
                    for entry in v]
                for k, v in self.metrics_history.items()
            }
        }, indent=2)
    
    def _export_csv_metrics(self) -> str:
        """Exporte les métriques au format CSV."""
        lines = ["timestamp,metric,value"]
        
        for metric_name, entries in self.metrics_history.items():
            for entry in entries:
                timestamp = entry["timestamp"].isoformat()
                value = entry["value"]
                lines.append(f"{timestamp},{metric_name},{value}")
        
        return "\n".join(lines)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Moniteur de configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python monitor_configs.py --namespace spotify-ai-agent-dev
  python monitor_configs.py --watch --interval 30
  python monitor_configs.py --export-metrics --format prometheus
        """
    )
    
    parser.add_argument(
        "--namespace", "-n",
        default="spotify-ai-agent-dev",
        help="Namespace Kubernetes à surveiller"
    )
    
    parser.add_argument(
        "--kubeconfig", "-k",
        help="Chemin vers le fichier kubeconfig"
    )
    
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Mode surveillance continue"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Intervalle de surveillance en secondes (défaut: 30)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        help="Durée de surveillance en secondes (illimitée par défaut)"
    )
    
    parser.add_argument(
        "--export-metrics",
        action="store_true",
        help="Exporte les métriques"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "prometheus", "csv"],
        default="json",
        help="Format d'export des métriques"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Fichier de sortie pour les métriques"
    )
    
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Vérification unique (pas de surveillance continue)"
    )
    
    args = parser.parse_args()
    
    try:
        # Création du moniteur
        monitor = ConfigurationMonitor(
            namespace=args.namespace,
            kubeconfig=args.kubeconfig,
            watch_interval=args.interval
        )
        
        # Vérification de la connectivité
        if not monitor.check_cluster_connectivity():
            print("❌ Impossible de se connecter au cluster Kubernetes")
            sys.exit(1)
        
        if args.one_shot:
            # Vérification unique
            status = monitor.get_configuration_status()
            monitor._display_status_summary(status)
            
            if status["alerts"]:
                monitor._display_alerts(status["alerts"])
            
        elif args.export_metrics:
            # Export des métriques
            if not monitor.last_check:
                # Collecte des données d'abord
                status = monitor.get_configuration_status()
                monitor.last_check = status
                monitor._save_metrics_history(status)
            
            metrics = monitor.export_metrics(args.format)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(metrics)
                print(f"📊 Métriques exportées vers {args.output}")
            else:
                print(metrics)
        
        elif args.watch:
            # Surveillance continue
            monitor.start_monitoring(args.duration)
        
        else:
            # Statut unique par défaut
            status = monitor.get_configuration_status()
            print(json.dumps(status, indent=2))
        
    except Exception as e:
        print(f"❌ Erreur lors de la surveillance: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
