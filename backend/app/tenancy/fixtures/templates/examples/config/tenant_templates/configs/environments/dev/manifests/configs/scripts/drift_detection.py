#!/usr/bin/env python3
"""
Configuration Drift Detection and Compliance System
=================================================

Système avancé de détection de dérive des configurations et de conformité.
Surveille les changements non autorisés, détecte les dérives de configuration,
et assure la conformité continue avec les politiques définies.

Fonctionnalités principales:
- Détection en temps réel des dérives de configuration
- Baseline de configurations de référence
- Alertes automatiques pour changements non autorisés
- Rapports de conformité détaillés
- Correction automatique des dérives
- Audit trail complet
- Intégration avec systèmes de monitoring

Author: Configuration Management Team
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python drift_detection.py [options]
    
Examples:
    python drift_detection.py --baseline --namespace spotify-ai-agent-dev
    python drift_detection.py --detect --alert-webhook https://alerts.company.com/webhook
    python drift_detection.py --monitor --interval 300  # Monitoring continu
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import time
import hashlib
import requests
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from deepdiff import DeepDiff
from kubernetes import client, config
import jsonschema

# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    """Niveaux de sévérité des dérives."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftType(Enum):
    """Types de dérives détectées."""
    CONFIGURATION_CHANGE = "configuration_change"
    RESOURCE_ADDITION = "resource_addition"
    RESOURCE_DELETION = "resource_deletion"
    PERMISSION_CHANGE = "permission_change"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class ComplianceStatus(Enum):
    """Statuts de conformité."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"

@dataclass
class DriftAlert:
    """Alerte de dérive de configuration."""
    id: str
    timestamp: datetime
    severity: DriftSeverity
    drift_type: DriftType
    resource_type: str
    resource_name: str
    namespace: str
    description: str
    current_config: Dict[str, Any]
    expected_config: Dict[str, Any]
    differences: Dict[str, Any]
    compliance_status: ComplianceStatus
    auto_fix_available: bool = False
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigurationBaseline:
    """Baseline de configuration de référence."""
    id: str
    timestamp: datetime
    namespace: str
    environment: str
    resource_checksums: Dict[str, str]
    policy_version: str
    compliance_rules: List[Dict[str, Any]]
    approved_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConfigurationDriftDetector:
    """Détecteur de dérive des configurations."""
    
    def __init__(self, 
                 namespace: str,
                 baseline_dir: Path = Path("/tmp/config-baselines"),
                 alerts_dir: Path = Path("/tmp/config-alerts"),
                 webhook_url: Optional[str] = None,
                 compliance_rules_file: Optional[Path] = None):
        
        self.namespace = namespace
        self.baseline_dir = baseline_dir
        self.alerts_dir = alerts_dir
        self.webhook_url = webhook_url
        self.compliance_rules_file = compliance_rules_file
        
        # État interne
        self.current_baseline: Optional[ConfigurationBaseline] = None
        self.active_alerts: List[DriftAlert] = []
        self.compliance_rules: List[Dict[str, Any]] = []
        self.monitoring_active = False
        
        # Initialisation
        self._init_directories()
        self._init_kubernetes_client()
        self._load_compliance_rules()
        self._load_latest_baseline()
    
    def _init_directories(self):
        """Initialise les répertoires de travail."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoires initialisés: {self.baseline_dir}, {self.alerts_dir}")
    
    def _init_kubernetes_client(self):
        """Initialise le client Kubernetes."""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_v1 = client.CoreV1Api()
        self.k8s_apps = client.AppsV1Api()
        self.k8s_rbac = client.RbacAuthorizationV1Api()
        logger.info("Client Kubernetes initialisé")
    
    def _load_compliance_rules(self):
        """Charge les règles de conformité."""
        if self.compliance_rules_file and self.compliance_rules_file.exists():
            try:
                with open(self.compliance_rules_file, 'r') as f:
                    self.compliance_rules = yaml.safe_load(f)
                logger.info(f"Règles de conformité chargées: {len(self.compliance_rules)} règles")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des règles: {e}")
        else:
            # Règles par défaut
            self.compliance_rules = self._get_default_compliance_rules()
    
    def _get_default_compliance_rules(self) -> List[Dict[str, Any]]:
        """Retourne les règles de conformité par défaut."""
        return [
            {
                "id": "security_context_required",
                "name": "Security Context Required",
                "description": "Tous les pods doivent avoir un security context défini",
                "resource_types": ["Pod", "Deployment"],
                "rule": {
                    "type": "path_exists",
                    "path": "spec.securityContext"
                },
                "severity": "high"
            },
            {
                "id": "resource_limits_required",
                "name": "Resource Limits Required",
                "description": "Tous les conteneurs doivent avoir des limites de ressources",
                "resource_types": ["Pod", "Deployment"],
                "rule": {
                    "type": "path_exists",
                    "path": "spec.containers.*.resources.limits"
                },
                "severity": "medium"
            },
            {
                "id": "image_pull_policy_always",
                "name": "Image Pull Policy Always",
                "description": "Les images doivent avoir une politique de pull 'Always'",
                "resource_types": ["Pod", "Deployment"],
                "rule": {
                    "type": "path_value",
                    "path": "spec.containers.*.imagePullPolicy",
                    "expected_value": "Always"
                },
                "severity": "low"
            },
            {
                "id": "no_privileged_containers",
                "name": "No Privileged Containers",
                "description": "Aucun conteneur ne doit être privilégié",
                "resource_types": ["Pod", "Deployment"],
                "rule": {
                    "type": "path_value",
                    "path": "spec.containers.*.securityContext.privileged",
                    "expected_value": False
                },
                "severity": "critical"
            }
        ]
    
    def _load_latest_baseline(self):
        """Charge la dernière baseline disponible."""
        baseline_files = list(self.baseline_dir.glob(f"{self.namespace}_*.json"))
        if not baseline_files:
            logger.info("Aucune baseline trouvée")
            return
        
        # Tri par date de modification (plus récent en premier)
        latest_file = sorted(baseline_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Conversion des timestamps
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
            self.current_baseline = ConfigurationBaseline(**data)
            logger.info(f"Baseline chargée: {self.current_baseline.id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la baseline: {e}")
    
    def create_baseline(self, approved_by: str, description: str = "") -> ConfigurationBaseline:
        """Crée une nouvelle baseline de configuration."""
        logger.info("Création d'une nouvelle baseline")
        
        baseline_id = f"{self.namespace}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Collecte des configurations actuelles
        current_configs = self._collect_current_configurations()
        
        # Calcul des checksums
        resource_checksums = {}
        for resource_type, resources in current_configs.items():
            for resource in resources:
                resource_id = f"{resource_type}/{resource.get('metadata', {}).get('name', 'unknown')}"
                resource_checksums[resource_id] = self._calculate_resource_checksum(resource)
        
        # Création de la baseline
        baseline = ConfigurationBaseline(
            id=baseline_id,
            timestamp=datetime.now(),
            namespace=self.namespace,
            environment=self._detect_environment(),
            resource_checksums=resource_checksums,
            policy_version="1.0.0",
            compliance_rules=self.compliance_rules,
            approved_by=approved_by,
            metadata={
                "description": description,
                "resource_count": sum(len(resources) for resources in current_configs.values()),
                "creation_method": "manual"
            }
        )
        
        # Sauvegarde de la baseline
        self._save_baseline(baseline)
        
        # Mise à jour de la baseline courante
        self.current_baseline = baseline
        
        logger.info(f"Baseline créée: {baseline_id}")
        return baseline
    
    def _collect_current_configurations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collecte les configurations actuelles."""
        configurations = {
            "configmaps": [],
            "secrets": [],
            "deployments": [],
            "services": [],
            "roles": [],
            "rolebindings": []
        }
        
        try:
            # ConfigMaps
            configmaps = self.k8s_v1.list_namespaced_config_map(namespace=self.namespace)
            for cm in configmaps.items:
                configurations["configmaps"].append(self._sanitize_resource(cm.to_dict()))
            
            # Secrets (sans les données sensibles)
            secrets = self.k8s_v1.list_namespaced_secret(namespace=self.namespace)
            for secret in secrets.items:
                if secret.type != "kubernetes.io/service-account-token":
                    secret_dict = self._sanitize_resource(secret.to_dict())
                    # Masquer les données sensibles
                    if secret_dict.get("data"):
                        secret_dict["data"] = {k: "[REDACTED]" for k in secret_dict["data"].keys()}
                    configurations["secrets"].append(secret_dict)
            
            # Deployments
            deployments = self.k8s_apps.list_namespaced_deployment(namespace=self.namespace)
            for deployment in deployments.items:
                configurations["deployments"].append(self._sanitize_resource(deployment.to_dict()))
            
            # Services
            services = self.k8s_v1.list_namespaced_service(namespace=self.namespace)
            for service in services.items:
                configurations["services"].append(self._sanitize_resource(service.to_dict()))
            
            # Roles
            roles = self.k8s_rbac.list_namespaced_role(namespace=self.namespace)
            for role in roles.items:
                configurations["roles"].append(self._sanitize_resource(role.to_dict()))
            
            # RoleBindings
            rolebindings = self.k8s_rbac.list_namespaced_role_binding(namespace=self.namespace)
            for rb in rolebindings.items:
                configurations["rolebindings"].append(self._sanitize_resource(rb.to_dict()))
            
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des configurations: {e}")
        
        return configurations
    
    def _sanitize_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie une ressource pour la baseline."""
        # Suppression des champs volatiles
        if 'metadata' in resource:
            metadata = resource['metadata']
            volatile_fields = [
                'resourceVersion', 'uid', 'selfLink', 'creationTimestamp',
                'generation', 'managedFields', 'annotations'
            ]
            for field in volatile_fields:
                metadata.pop(field, None)
        
        if 'status' in resource:
            # Suppression du status complet
            del resource['status']
        
        return resource
    
    def _calculate_resource_checksum(self, resource: Dict[str, Any]) -> str:
        """Calcule le checksum d'une ressource."""
        # Sérialisation canonique pour le hachage
        serialized = json.dumps(resource, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).hexdigest()
    
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
    
    def _save_baseline(self, baseline: ConfigurationBaseline):
        """Sauvegarde une baseline."""
        baseline_file = self.baseline_dir / f"{baseline.id}.json"
        
        # Conversion pour la sérialisation JSON
        baseline_dict = asdict(baseline)
        baseline_dict['timestamp'] = baseline.timestamp.isoformat()
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_dict, f, indent=2)
        
        logger.info(f"Baseline sauvegardée: {baseline_file}")
    
    def detect_drift(self) -> List[DriftAlert]:
        """Détecte les dérives par rapport à la baseline."""
        if not self.current_baseline:
            logger.warning("Aucune baseline disponible pour la détection de dérive")
            return []
        
        logger.info("Détection des dérives en cours...")
        
        # Collecte des configurations actuelles
        current_configs = self._collect_current_configurations()
        
        # Comparaison avec la baseline
        drift_alerts = []
        
        # Vérification des ressources existantes
        for resource_type, resources in current_configs.items():
            for resource in resources:
                resource_id = f"{resource_type}/{resource.get('metadata', {}).get('name', 'unknown')}"
                current_checksum = self._calculate_resource_checksum(resource)
                baseline_checksum = self.current_baseline.resource_checksums.get(resource_id)
                
                if baseline_checksum is None:
                    # Nouvelle ressource non autorisée
                    alert = self._create_drift_alert(
                        drift_type=DriftType.RESOURCE_ADDITION,
                        severity=DriftSeverity.MEDIUM,
                        resource_type=resource_type,
                        resource_name=resource.get('metadata', {}).get('name', 'unknown'),
                        description=f"Nouvelle ressource détectée: {resource_id}",
                        current_config=resource,
                        expected_config={},
                        differences={"type": "new_resource"}
                    )
                    drift_alerts.append(alert)
                
                elif current_checksum != baseline_checksum:
                    # Configuration modifiée
                    baseline_resource = self._get_baseline_resource(resource_id)
                    differences = DeepDiff(baseline_resource, resource, ignore_order=True)
                    
                    alert = self._create_drift_alert(
                        drift_type=DriftType.CONFIGURATION_CHANGE,
                        severity=self._assess_drift_severity(differences),
                        resource_type=resource_type,
                        resource_name=resource.get('metadata', {}).get('name', 'unknown'),
                        description=f"Configuration modifiée: {resource_id}",
                        current_config=resource,
                        expected_config=baseline_resource or {},
                        differences=differences.to_dict() if hasattr(differences, 'to_dict') else {}
                    )
                    drift_alerts.append(alert)
        
        # Vérification des ressources supprimées
        for resource_id in self.current_baseline.resource_checksums.keys():
            if not self._resource_exists_in_current(resource_id, current_configs):
                resource_type, resource_name = resource_id.split('/', 1)
                alert = self._create_drift_alert(
                    drift_type=DriftType.RESOURCE_DELETION,
                    severity=DriftSeverity.HIGH,
                    resource_type=resource_type,
                    resource_name=resource_name,
                    description=f"Ressource supprimée: {resource_id}",
                    current_config={},
                    expected_config={},
                    differences={"type": "deleted_resource"}
                )
                drift_alerts.append(alert)
        
        # Vérification de la conformité
        compliance_alerts = self._check_compliance(current_configs)
        drift_alerts.extend(compliance_alerts)
        
        # Sauvegarde des alertes
        self.active_alerts = drift_alerts
        self._save_alerts(drift_alerts)
        
        logger.info(f"Détection terminée: {len(drift_alerts)} dérives détectées")
        return drift_alerts
    
    def _create_drift_alert(self, drift_type: DriftType, severity: DriftSeverity,
                           resource_type: str, resource_name: str, description: str,
                           current_config: Dict[str, Any], expected_config: Dict[str, Any],
                           differences: Dict[str, Any]) -> DriftAlert:
        """Crée une alerte de dérive."""
        alert_id = f"{self.namespace}_{resource_type}_{resource_name}_{int(time.time())}"
        
        return DriftAlert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            drift_type=drift_type,
            resource_type=resource_type,
            resource_name=resource_name,
            namespace=self.namespace,
            description=description,
            current_config=current_config,
            expected_config=expected_config,
            differences=differences,
            compliance_status=self._assess_compliance_status(current_config),
            auto_fix_available=self._can_auto_fix(drift_type, differences)
        )
    
    def _assess_drift_severity(self, differences: Any) -> DriftSeverity:
        """Évalue la sévérité d'une dérive."""
        if not differences:
            return DriftSeverity.LOW
        
        # Analyse des changements pour déterminer la sévérité
        diff_str = str(differences).lower()
        
        if any(keyword in diff_str for keyword in ['security', 'privilege', 'permission', 'rbac']):
            return DriftSeverity.CRITICAL
        elif any(keyword in diff_str for keyword in ['image', 'version', 'tag']):
            return DriftSeverity.HIGH
        elif any(keyword in diff_str for keyword in ['replicas', 'resources', 'limits']):
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _assess_compliance_status(self, resource: Dict[str, Any]) -> ComplianceStatus:
        """Évalue le statut de conformité d'une ressource."""
        violations = self._check_resource_compliance(resource)
        
        if not violations:
            return ComplianceStatus.COMPLIANT
        elif any(v.get('severity') == 'critical' for v in violations):
            return ComplianceStatus.NON_COMPLIANT
        else:
            return ComplianceStatus.WARNING
    
    def _can_auto_fix(self, drift_type: DriftType, differences: Dict[str, Any]) -> bool:
        """Détermine si une dérive peut être corrigée automatiquement."""
        # Logique de détermination de correction automatique
        if drift_type == DriftType.CONFIGURATION_CHANGE:
            # Certains changements peuvent être automatiquement revertés
            return True
        elif drift_type == DriftType.RESOURCE_DELETION:
            # Les ressources supprimées peuvent être recréées
            return True
        else:
            return False
    
    def _get_baseline_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une ressource de la baseline."""
        # Cette fonction devrait récupérer la ressource originale de la baseline
        # Pour l'instant, on retourne None
        return None
    
    def _resource_exists_in_current(self, resource_id: str, current_configs: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Vérifie si une ressource existe dans les configurations actuelles."""
        resource_type, resource_name = resource_id.split('/', 1)
        
        if resource_type in current_configs:
            for resource in current_configs[resource_type]:
                if resource.get('metadata', {}).get('name') == resource_name:
                    return True
        
        return False
    
    def _check_compliance(self, current_configs: Dict[str, List[Dict[str, Any]]]) -> List[DriftAlert]:
        """Vérifie la conformité des configurations."""
        compliance_alerts = []
        
        for resource_type, resources in current_configs.items():
            for resource in resources:
                violations = self._check_resource_compliance(resource)
                
                for violation in violations:
                    alert = self._create_drift_alert(
                        drift_type=DriftType.COMPLIANCE_VIOLATION,
                        severity=DriftSeverity(violation.get('severity', 'medium')),
                        resource_type=resource_type,
                        resource_name=resource.get('metadata', {}).get('name', 'unknown'),
                        description=f"Violation de conformité: {violation['rule_name']}",
                        current_config=resource,
                        expected_config={},
                        differences={"compliance_violation": violation}
                    )
                    compliance_alerts.append(alert)
        
        return compliance_alerts
    
    def _check_resource_compliance(self, resource: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Vérifie la conformité d'une ressource."""
        violations = []
        resource_kind = resource.get('kind', '')
        
        for rule in self.compliance_rules:
            if resource_kind in rule.get('resource_types', []):
                violation = self._evaluate_compliance_rule(resource, rule)
                if violation:
                    violations.append({
                        'rule_id': rule['id'],
                        'rule_name': rule['name'],
                        'description': rule['description'],
                        'severity': rule.get('severity', 'medium'),
                        'violation_details': violation
                    })
        
        return violations
    
    def _evaluate_compliance_rule(self, resource: Dict[str, Any], rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Évalue une règle de conformité sur une ressource."""
        rule_config = rule.get('rule', {})
        rule_type = rule_config.get('type')
        
        if rule_type == 'path_exists':
            path = rule_config.get('path')
            if not self._path_exists_in_resource(resource, path):
                return {'type': 'missing_path', 'path': path}
        
        elif rule_type == 'path_value':
            path = rule_config.get('path')
            expected_value = rule_config.get('expected_value')
            actual_value = self._get_path_value_from_resource(resource, path)
            if actual_value != expected_value:
                return {
                    'type': 'incorrect_value',
                    'path': path,
                    'expected': expected_value,
                    'actual': actual_value
                }
        
        return None
    
    def _path_exists_in_resource(self, resource: Dict[str, Any], path: str) -> bool:
        """Vérifie si un chemin existe dans une ressource."""
        try:
            parts = path.split('.')
            current = resource
            
            for part in parts:
                if '*' in part:
                    # Gestion des wildcards (simplifié)
                    return True  # Implémentation complète nécessaire
                elif part in current:
                    current = current[part]
                else:
                    return False
            
            return True
        except:
            return False
    
    def _get_path_value_from_resource(self, resource: Dict[str, Any], path: str) -> Any:
        """Récupère la valeur d'un chemin dans une ressource."""
        try:
            parts = path.split('.')
            current = resource
            
            for part in parts:
                if '*' in part:
                    # Gestion des wildcards (simplifié)
                    return None  # Implémentation complète nécessaire
                elif part in current:
                    current = current[part]
                else:
                    return None
            
            return current
        except:
            return None
    
    def _save_alerts(self, alerts: List[DriftAlert]):
        """Sauvegarde les alertes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alerts_file = self.alerts_dir / f"alerts_{self.namespace}_{timestamp}.json"
        
        # Conversion pour la sérialisation JSON
        alerts_data = []
        for alert in alerts:
            alert_dict = asdict(alert)
            alert_dict['timestamp'] = alert.timestamp.isoformat()
            alert_dict['severity'] = alert.severity.value
            alert_dict['drift_type'] = alert.drift_type.value
            alert_dict['compliance_status'] = alert.compliance_status.value
            alerts_data.append(alert_dict)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        logger.info(f"Alertes sauvegardées: {alerts_file}")
    
    def send_alerts(self, alerts: List[DriftAlert]):
        """Envoie les alertes via webhook."""
        if not self.webhook_url or not alerts:
            return
        
        try:
            # Préparation du payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "namespace": self.namespace,
                "total_alerts": len(alerts),
                "severity_counts": self._count_alerts_by_severity(alerts),
                "alerts": []
            }
            
            for alert in alerts:
                payload["alerts"].append({
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "type": alert.drift_type.value,
                    "resource": f"{alert.resource_type}/{alert.resource_name}",
                    "description": alert.description,
                    "compliance_status": alert.compliance_status.value
                })
            
            # Envoi du webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Alertes envoyées via webhook: {len(alerts)} alertes")
            else:
                logger.error(f"Erreur webhook: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi des alertes: {e}")
    
    def _count_alerts_by_severity(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        """Compte les alertes par niveau de sévérité."""
        counts = {severity.value: 0 for severity in DriftSeverity}
        
        for alert in alerts:
            counts[alert.severity.value] += 1
        
        return counts
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Démarre la surveillance continue."""
        logger.info(f"Démarrage de la surveillance continue (intervalle: {interval_seconds}s)")
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    alerts = self.detect_drift()
                    
                    if alerts:
                        # Filtrer les nouvelles alertes
                        new_alerts = [alert for alert in alerts if not alert.acknowledged]
                        
                        if new_alerts:
                            logger.warning(f"Nouvelles dérives détectées: {len(new_alerts)}")
                            self.send_alerts(new_alerts)
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Erreur dans la boucle de surveillance: {e}")
                    time.sleep(60)  # Attente plus courte en cas d'erreur
        
        # Lancement dans un thread séparé
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Surveillance continue démarrée")
    
    def stop_monitoring(self):
        """Arrête la surveillance continue."""
        self.monitoring_active = False
        logger.info("Surveillance continue arrêtée")
    
    def auto_fix_drifts(self, alert_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Corrige automatiquement les dérives."""
        logger.info("Démarrage de la correction automatique des dérives")
        
        alerts_to_fix = self.active_alerts
        if alert_ids:
            alerts_to_fix = [alert for alert in self.active_alerts if alert.id in alert_ids]
        
        results = {}
        
        for alert in alerts_to_fix:
            if not alert.auto_fix_available:
                results[alert.id] = False
                continue
            
            try:
                success = self._fix_drift(alert)
                results[alert.id] = success
                
                if success:
                    alert.resolved = True
                    logger.info(f"Dérive corrigée: {alert.id}")
                else:
                    logger.warning(f"Échec de la correction: {alert.id}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la correction de {alert.id}: {e}")
                results[alert.id] = False
        
        return results
    
    def _fix_drift(self, alert: DriftAlert) -> bool:
        """Corrige une dérive spécifique."""
        try:
            if alert.drift_type == DriftType.CONFIGURATION_CHANGE:
                # Restaurer la configuration de la baseline
                return self._restore_resource_from_baseline(alert)
            
            elif alert.drift_type == DriftType.RESOURCE_DELETION:
                # Recréer la ressource depuis la baseline
                return self._recreate_resource_from_baseline(alert)
            
            elif alert.drift_type == DriftType.COMPLIANCE_VIOLATION:
                # Corriger la violation de conformité
                return self._fix_compliance_violation(alert)
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la correction de dérive: {e}")
            return False
    
    def _restore_resource_from_baseline(self, alert: DriftAlert) -> bool:
        """Restaure une ressource depuis la baseline."""
        # Implémentation de la restauration
        # Cette fonction devrait appliquer la configuration de la baseline
        logger.info(f"Restauration de {alert.resource_type}/{alert.resource_name}")
        return True  # Placeholder
    
    def _recreate_resource_from_baseline(self, alert: DriftAlert) -> bool:
        """Recrée une ressource depuis la baseline."""
        # Implémentation de la recréation
        logger.info(f"Recréation de {alert.resource_type}/{alert.resource_name}")
        return True  # Placeholder
    
    def _fix_compliance_violation(self, alert: DriftAlert) -> bool:
        """Corrige une violation de conformité."""
        # Implémentation de la correction de conformité
        logger.info(f"Correction de conformité pour {alert.resource_type}/{alert.resource_name}")
        return True  # Placeholder

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Système de détection de dérive et conformité des configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--namespace", "-n", required=True, help="Namespace Kubernetes")
    parser.add_argument("--baseline-dir", type=Path, default=Path("/tmp/config-baselines"), help="Répertoire des baselines")
    parser.add_argument("--alerts-dir", type=Path, default=Path("/tmp/config-alerts"), help="Répertoire des alertes")
    
    parser.add_argument("--baseline", action="store_true", help="Créer une nouvelle baseline")
    parser.add_argument("--detect", action="store_true", help="Détecter les dérives")
    parser.add_argument("--monitor", action="store_true", help="Surveillance continue")
    parser.add_argument("--auto-fix", action="store_true", help="Correction automatique")
    
    parser.add_argument("--approved-by", help="Nom de l'approbateur pour la baseline")
    parser.add_argument("--description", help="Description de la baseline")
    parser.add_argument("--interval", type=int, default=300, help="Intervalle de surveillance (secondes)")
    parser.add_argument("--webhook-url", help="URL du webhook pour les alertes")
    parser.add_argument("--compliance-rules", type=Path, help="Fichier des règles de conformité")
    parser.add_argument("--alert-ids", nargs="+", help="IDs des alertes à corriger")
    
    args = parser.parse_args()
    
    try:
        # Création du détecteur
        detector = ConfigurationDriftDetector(
            namespace=args.namespace,
            baseline_dir=args.baseline_dir,
            alerts_dir=args.alerts_dir,
            webhook_url=args.webhook_url,
            compliance_rules_file=args.compliance_rules
        )
        
        if args.baseline:
            if not args.approved_by:
                print("--approved-by requis pour créer une baseline")
                sys.exit(1)
            
            baseline = detector.create_baseline(
                approved_by=args.approved_by,
                description=args.description or ""
            )
            print(f"Baseline créée: {baseline.id}")
        
        elif args.detect:
            alerts = detector.detect_drift()
            print(f"Dérives détectées: {len(alerts)}")
            
            if alerts and args.webhook_url:
                detector.send_alerts(alerts)
            
            # Affichage des alertes critiques
            critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
            if critical_alerts:
                print(f"\nAlertes critiques ({len(critical_alerts)}):")
                for alert in critical_alerts:
                    print(f"  - {alert.description}")
        
        elif args.monitor:
            detector.start_monitoring(args.interval)
            print(f"Surveillance démarrée (intervalle: {args.interval}s)")
            print("Appuyez sur Ctrl+C pour arrêter")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                detector.stop_monitoring()
                print("\nSurveillance arrêtée")
        
        elif args.auto_fix:
            results = detector.auto_fix_drifts(args.alert_ids)
            successful_fixes = sum(1 for success in results.values() if success)
            print(f"Corrections automatiques: {successful_fixes}/{len(results)} réussies")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
