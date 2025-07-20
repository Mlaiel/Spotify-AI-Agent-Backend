#!/usr/bin/env python3
"""
Configuration Security Scanner
=============================

Scanner de sécurité avancé pour les configurations Kubernetes.
Détecte les vulnérabilités, mauvaises pratiques et non-conformités.

Author: Security Engineering Team - Spotify AI Agent
Team: Security & Compliance Division
Version: 2.0.0
Date: July 17, 2025

Usage:
    python security_scanner.py [options]
    
Examples:
    python security_scanner.py --namespace spotify-ai-agent-dev --full-scan
    python security_scanner.py --config-dir ./configs/ --compliance-check
    python security_scanner.py --export-report --format sarif
"""

import argparse
import json
import yaml
import os
import sys
import subprocess
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SeverityLevel(Enum):
    """Niveaux de sévérité des problèmes de sécurité."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceStandard(Enum):
    """Standards de conformité supportés."""
    CIS_KUBERNETES = "cis-kubernetes"
    NIST_CYBERSECURITY = "nist-cybersecurity"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci-dss"

@dataclass
class SecurityFinding:
    """Résultat d'analyse de sécurité."""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    category: str
    resource_type: str
    resource_name: str
    namespace: str
    rule_id: str
    compliance_standards: List[ComplianceStandard]
    remediation: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[Dict[str, Any]] = None
    cve_references: Optional[List[str]] = None
    cvss_score: Optional[float] = None

@dataclass
class SecurityReport:
    """Rapport de sécurité complet."""
    scan_id: str
    timestamp: datetime
    namespace: str
    total_resources: int
    findings: List[SecurityFinding]
    compliance_scores: Dict[str, float]
    risk_score: float
    scan_duration: float
    recommendations: List[str]

class SecurityScanner:
    """Scanner de sécurité avancé pour configurations Kubernetes."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 kubeconfig: Optional[str] = None,
                 config_dir: Optional[Path] = None):
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.config_dir = config_dir
        self.findings = []
        self.scanned_resources = 0
        self.scan_start_time = None
        
        # Chargement des règles de sécurité
        self.security_rules = self._load_security_rules()
        self.compliance_rules = self._load_compliance_rules()
        
        # Patterns de détection
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.vulnerability_db = self._load_vulnerability_database()
    
    def scan_configurations(self, scan_types: List[str] = None) -> SecurityReport:
        """Lance un scan complet de sécurité."""
        print("🔍 Démarrage du scan de sécurité...")
        
        self.scan_start_time = datetime.now()
        scan_id = f"scan-{self.scan_start_time.strftime('%Y%m%d-%H%M%S')}"
        
        # Types de scan par défaut
        if not scan_types:
            scan_types = ["configuration", "secrets", "rbac", "network", "compliance"]
        
        try:
            # Récupération des ressources
            if self.config_dir:
                resources = self._load_local_configurations()
            else:
                resources = self._load_cluster_configurations()
            
            self.scanned_resources = len(resources)
            print(f"📊 Analyse de {self.scanned_resources} ressources...")
            
            # Exécution des scans
            for scan_type in scan_types:
                print(f"   🔍 Scan {scan_type}...")
                self._execute_scan_type(scan_type, resources)
            
            # Calculs des scores
            compliance_scores = self._calculate_compliance_scores()
            risk_score = self._calculate_risk_score()
            recommendations = self._generate_recommendations()
            
            # Génération du rapport
            scan_duration = (datetime.now() - self.scan_start_time).total_seconds()
            
            report = SecurityReport(
                scan_id=scan_id,
                timestamp=self.scan_start_time,
                namespace=self.namespace,
                total_resources=self.scanned_resources,
                findings=self.findings,
                compliance_scores=compliance_scores,
                risk_score=risk_score,
                scan_duration=scan_duration,
                recommendations=recommendations
            )
            
            print(f"✅ Scan terminé: {len(self.findings)} problèmes détectés")
            print(f"   Score de risque: {risk_score:.1f}/100")
            
            return report
            
        except Exception as e:
            print(f"❌ Erreur lors du scan: {e}")
            raise
    
    def _execute_scan_type(self, scan_type: str, resources: List[Dict[str, Any]]) -> None:
        """Exécute un type de scan spécifique."""
        if scan_type == "configuration":
            self._scan_configuration_security(resources)
        elif scan_type == "secrets":
            self._scan_secrets_security(resources)
        elif scan_type == "rbac":
            self._scan_rbac_security(resources)
        elif scan_type == "network":
            self._scan_network_security(resources)
        elif scan_type == "compliance":
            self._scan_compliance(resources)
        elif scan_type == "vulnerabilities":
            self._scan_vulnerabilities(resources)
        elif scan_type == "best_practices":
            self._scan_best_practices(resources)
    
    def _scan_configuration_security(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de sécurité des configurations."""
        for resource in resources:
            kind = resource.get("kind", "")
            metadata = resource.get("metadata", {})
            spec = resource.get("spec", {})
            
            # Vérifications génériques
            self._check_resource_labels(resource)
            self._check_resource_annotations(resource)
            
            # Vérifications spécifiques par type
            if kind == "Pod" or (kind == "Deployment" and "template" in spec):
                self._check_pod_security(resource)
            elif kind == "Service":
                self._check_service_security(resource)
            elif kind == "Ingress":
                self._check_ingress_security(resource)
            elif kind == "ConfigMap":
                self._check_configmap_security(resource)
    
    def _scan_secrets_security(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de sécurité des secrets."""
        for resource in resources:
            if resource.get("kind") == "Secret":
                self._check_secret_security(resource)
            
            # Recherche de secrets hardcodés dans d'autres ressources
            self._check_hardcoded_secrets(resource)
    
    def _scan_rbac_security(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de sécurité RBAC."""
        for resource in resources:
            kind = resource.get("kind", "")
            
            if kind == "Role":
                self._check_role_security(resource)
            elif kind == "ClusterRole":
                self._check_cluster_role_security(resource)
            elif kind == "RoleBinding":
                self._check_role_binding_security(resource)
            elif kind == "ClusterRoleBinding":
                self._check_cluster_role_binding_security(resource)
            elif kind == "ServiceAccount":
                self._check_service_account_security(resource)
    
    def _scan_network_security(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de sécurité réseau."""
        for resource in resources:
            kind = resource.get("kind", "")
            
            if kind == "NetworkPolicy":
                self._check_network_policy_security(resource)
            elif kind == "Service":
                self._check_service_exposure(resource)
            elif kind == "Ingress":
                self._check_ingress_exposure(resource)
    
    def _scan_compliance(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de conformité réglementaire."""
        for standard in ComplianceStandard:
            self._check_compliance_standard(resources, standard)
    
    def _scan_vulnerabilities(self, resources: List[Dict[str, Any]]) -> None:
        """Scan de vulnérabilités connues."""
        for resource in resources:
            self._check_known_vulnerabilities(resource)
    
    def _scan_best_practices(self, resources: List[Dict[str, Any]]) -> None:
        """Scan des meilleures pratiques."""
        for resource in resources:
            self._check_security_best_practices(resource)
    
    # Méthodes de vérification spécifiques
    
    def _check_pod_security(self, resource: Dict[str, Any]) -> None:
        """Vérifications de sécurité pour les pods."""
        pod_spec = resource.get("spec", {})
        
        # Extraction du template pour les Deployments
        if resource.get("kind") == "Deployment":
            pod_spec = pod_spec.get("template", {}).get("spec", {})
        
        # Vérification du contexte de sécurité
        security_context = pod_spec.get("securityContext", {})
        
        # runAsNonRoot
        if not security_context.get("runAsNonRoot", False):
            self._add_finding(
                id="POD-001",
                title="Pod s'exécute en tant que root",
                description="Le pod peut s'exécuter avec des privilèges root",
                severity=SeverityLevel.HIGH,
                category="pod_security",
                resource=resource,
                rule_id="runAsNonRoot",
                compliance=[ComplianceStandard.CIS_KUBERNETES, ComplianceStandard.NIST_CYBERSECURITY],
                remediation="Définir securityContext.runAsNonRoot: true"
            )
        
        # readOnlyRootFilesystem
        for container in pod_spec.get("containers", []):
            container_security = container.get("securityContext", {})
            if not container_security.get("readOnlyRootFilesystem", False):
                self._add_finding(
                    id="POD-002",
                    title="Système de fichiers racine non en lecture seule",
                    description=f"Le conteneur '{container.get('name')}' n'a pas un système de fichiers racine en lecture seule",
                    severity=SeverityLevel.MEDIUM,
                    category="pod_security",
                    resource=resource,
                    rule_id="readOnlyRootFilesystem",
                    compliance=[ComplianceStandard.CIS_KUBERNETES],
                    remediation="Définir securityContext.readOnlyRootFilesystem: true"
                )
            
            # Privilèges élevés
            if container_security.get("privileged", False):
                self._add_finding(
                    id="POD-003",
                    title="Conteneur privilégié détecté",
                    description=f"Le conteneur '{container.get('name')}' s'exécute en mode privilégié",
                    severity=SeverityLevel.CRITICAL,
                    category="pod_security",
                    resource=resource,
                    rule_id="privileged",
                    compliance=[ComplianceStandard.CIS_KUBERNETES, ComplianceStandard.NIST_CYBERSECURITY],
                    remediation="Retirer securityContext.privileged ou définir à false"
                )
            
            # Capacités dangereuses
            capabilities = container_security.get("capabilities", {})
            dangerous_caps = {"SYS_ADMIN", "NET_ADMIN", "SYS_TIME", "NET_RAW"}
            added_caps = set(capabilities.get("add", []))
            
            if dangerous_caps.intersection(added_caps):
                dangerous_found = dangerous_caps.intersection(added_caps)
                self._add_finding(
                    id="POD-004",
                    title="Capacités dangereuses ajoutées",
                    description=f"Le conteneur '{container.get('name')}' ajoute des capacités dangereuses: {', '.join(dangerous_found)}",
                    severity=SeverityLevel.HIGH,
                    category="pod_security",
                    resource=resource,
                    rule_id="dangerous_capabilities",
                    compliance=[ComplianceStandard.CIS_KUBERNETES],
                    remediation="Retirer les capacités dangereuses ou utiliser drop: [ALL]"
                )
        
        # Volumes hostPath
        for volume in pod_spec.get("volumes", []):
            if "hostPath" in volume:
                self._add_finding(
                    id="POD-005",
                    title="Volume hostPath détecté",
                    description=f"Le volume '{volume.get('name')}' utilise hostPath: {volume['hostPath'].get('path')}",
                    severity=SeverityLevel.HIGH,
                    category="pod_security",
                    resource=resource,
                    rule_id="hostPath_volume",
                    compliance=[ComplianceStandard.CIS_KUBERNETES],
                    remediation="Utiliser des volumes persistants ou des ConfigMaps/Secrets à la place"
                )
    
    def _check_secret_security(self, resource: Dict[str, Any]) -> None:
        """Vérifications de sécurité pour les secrets."""
        metadata = resource.get("metadata", {})
        data = resource.get("data", {})
        
        # Vérification du chiffrement
        if not self._is_secret_encrypted_at_rest():
            self._add_finding(
                id="SECRET-001",
                title="Secret non chiffré au repos",
                description="Le secret n'est pas chiffré au repos dans etcd",
                severity=SeverityLevel.HIGH,
                category="secret_security",
                resource=resource,
                rule_id="encryption_at_rest",
                compliance=[ComplianceStandard.GDPR, ComplianceStandard.SOC2],
                remediation="Activer le chiffrement etcd au niveau du cluster"
            )
        
        # Analyse du contenu des secrets
        for key, value in data.items():
            try:
                decoded_value = base64.b64decode(value).decode('utf-8')
                
                # Détection de patterns sensibles
                if self._contains_sensitive_patterns(decoded_value):
                    self._add_finding(
                        id="SECRET-002",
                        title="Contenu sensible détecté dans le secret",
                        description=f"La clé '{key}' contient potentiellement des données sensibles",
                        severity=SeverityLevel.MEDIUM,
                        category="secret_security",
                        resource=resource,
                        rule_id="sensitive_content",
                        compliance=[ComplianceStandard.GDPR],
                        remediation="Vérifier le contenu et utiliser des références externes si possible"
                    )
                
                # Vérification de la complexité des mots de passe
                if any(keyword in key.lower() for keyword in ["password", "pass", "pwd"]):
                    if not self._is_strong_password(decoded_value):
                        self._add_finding(
                            id="SECRET-003",
                            title="Mot de passe faible détecté",
                            description=f"La clé '{key}' contient un mot de passe faible",
                            severity=SeverityLevel.MEDIUM,
                            category="secret_security",
                            resource=resource,
                            rule_id="weak_password",
                            compliance=[ComplianceStandard.NIST_CYBERSECURITY],
                            remediation="Utiliser un mot de passe plus complexe (12+ caractères, mixte)"
                        )
                
            except Exception:
                # Impossible de décoder, probablement binaire
                continue
    
    def _check_hardcoded_secrets(self, resource: Dict[str, Any]) -> None:
        """Recherche de secrets hardcodés."""
        resource_str = json.dumps(resource)
        
        # Patterns de secrets hardcodés
        patterns = {
            "api_key": r"(?i)(api[_-]?key|apikey)[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{20,}",
            "password": r"(?i)(password|passwd|pwd)[\"'\s]*[:=][\"'\s]*[^\s\"']{8,}",
            "token": r"(?i)(token|jwt)[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9\.\-_]{20,}",
            "private_key": r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
            "connection_string": r"(?i)(connection[_-]?string|conn[_-]?str)[\"'\s]*[:=][\"'\s]*[^\s\"']+",
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, resource_str)
            if matches:
                self._add_finding(
                    id="HARDCODED-001",
                    title=f"Secret potentiel hardcodé détecté ({pattern_name})",
                    description=f"Pattern de {pattern_name} détecté dans la configuration",
                    severity=SeverityLevel.HIGH,
                    category="hardcoded_secrets",
                    resource=resource,
                    rule_id="hardcoded_secrets",
                    compliance=[ComplianceStandard.GDPR, ComplianceStandard.SOC2],
                    remediation="Déplacer les secrets vers des Kubernetes Secrets ou un gestionnaire de secrets externe",
                    evidence={"pattern": pattern_name, "matches_count": len(matches)}
                )
    
    def _check_rbac_security(self, resource: Dict[str, Any]) -> None:
        """Vérifications RBAC génériques."""
        kind = resource.get("kind", "")
        rules = resource.get("rules", [])
        
        # Vérification des permissions dangereuses
        dangerous_verbs = {"*", "create", "delete", "deletecollection"}
        dangerous_resources = {"*", "secrets", "nodes", "pods/exec", "pods/portforward"}
        
        for rule in rules:
            verbs = set(rule.get("verbs", []))
            resources = set(rule.get("resources", []))
            
            # Permissions wildcard
            if "*" in verbs and "*" in resources:
                self._add_finding(
                    id="RBAC-001",
                    title="Permissions wildcard détectées",
                    description="Règle RBAC avec permissions wildcard (*/*)",
                    severity=SeverityLevel.CRITICAL,
                    category="rbac_security",
                    resource=resource,
                    rule_id="wildcard_permissions",
                    compliance=[ComplianceStandard.CIS_KUBERNETES, ComplianceStandard.SOC2],
                    remediation="Spécifier des permissions granulaires au lieu d'utiliser des wildcards"
                )
            
            # Accès aux secrets
            if "secrets" in resources and dangerous_verbs.intersection(verbs):
                self._add_finding(
                    id="RBAC-002",
                    title="Accès étendu aux secrets",
                    description="Règle RBAC permettant la création/suppression de secrets",
                    severity=SeverityLevel.HIGH,
                    category="rbac_security",
                    resource=resource,
                    rule_id="secrets_access",
                    compliance=[ComplianceStandard.CIS_KUBERNETES],
                    remediation="Limiter l'accès aux secrets aux opérations strictement nécessaires"
                )
    
    def _check_network_policy_security(self, resource: Dict[str, Any]) -> None:
        """Vérifications des politiques réseau."""
        spec = resource.get("spec", {})
        
        # Vérification de la restriction du trafic
        if not spec.get("ingress") and not spec.get("egress"):
            self._add_finding(
                id="NETWORK-001",
                title="Politique réseau trop permissive",
                description="NetworkPolicy sans restrictions d'ingress ou egress",
                severity=SeverityLevel.MEDIUM,
                category="network_security",
                resource=resource,
                rule_id="permissive_network_policy",
                compliance=[ComplianceStandard.CIS_KUBERNETES],
                remediation="Définir des règles d'ingress et egress spécifiques"
            )
        
        # Vérification des règles ouvertes
        for ingress_rule in spec.get("ingress", []):
            if not ingress_rule.get("from"):
                self._add_finding(
                    id="NETWORK-002",
                    title="Règle d'ingress ouverte",
                    description="Règle d'ingress sans restriction de source",
                    severity=SeverityLevel.HIGH,
                    category="network_security",
                    resource=resource,
                    rule_id="open_ingress",
                    compliance=[ComplianceStandard.CIS_KUBERNETES],
                    remediation="Spécifier des sources autorisées pour les règles d'ingress"
                )
    
    def _check_compliance_standard(self, resources: List[Dict[str, Any]], 
                                 standard: ComplianceStandard) -> None:
        """Vérifications spécifiques à un standard de conformité."""
        if standard == ComplianceStandard.GDPR:
            self._check_gdpr_compliance(resources)
        elif standard == ComplianceStandard.CIS_KUBERNETES:
            self._check_cis_compliance(resources)
        elif standard == ComplianceStandard.SOC2:
            self._check_soc2_compliance(resources)
    
    def _check_gdpr_compliance(self, resources: List[Dict[str, Any]]) -> None:
        """Vérifications GDPR."""
        for resource in resources:
            # Vérification des annotations de données personnelles
            annotations = resource.get("metadata", {}).get("annotations", {})
            
            if any("personal" in key.lower() or "gdpr" in key.lower() for key in annotations.keys()):
                if not annotations.get("data-protection.gdpr.eu/retention-period"):
                    self._add_finding(
                        id="GDPR-001",
                        title="Période de rétention GDPR manquante",
                        description="Ressource contenant des données personnelles sans période de rétention définie",
                        severity=SeverityLevel.HIGH,
                        category="gdpr_compliance",
                        resource=resource,
                        rule_id="gdpr_retention",
                        compliance=[ComplianceStandard.GDPR],
                        remediation="Ajouter l'annotation data-protection.gdpr.eu/retention-period"
                    )
    
    def _check_cis_compliance(self, resources: List[Dict[str, Any]]) -> None:
        """Vérifications CIS Kubernetes Benchmark."""
        # Ces vérifications sont déjà intégrées dans les autres méthodes
        # Cette méthode peut être étendue pour des vérifications CIS spécifiques
        pass
    
    def _check_soc2_compliance(self, resources: List[Dict[str, Any]]) -> None:
        """Vérifications SOC2."""
        for resource in resources:
            # Vérification des logs d'audit
            if resource.get("kind") in ["Deployment", "StatefulSet", "DaemonSet"]:
                annotations = resource.get("metadata", {}).get("annotations", {})
                
                if not annotations.get("audit.soc2.com/enabled"):
                    self._add_finding(
                        id="SOC2-001",
                        title="Audit SOC2 non activé",
                        description="Ressource critique sans audit SOC2 activé",
                        severity=SeverityLevel.MEDIUM,
                        category="soc2_compliance",
                        resource=resource,
                        rule_id="soc2_audit",
                        compliance=[ComplianceStandard.SOC2],
                        remediation="Ajouter l'annotation audit.soc2.com/enabled: 'true'"
                    )
    
    # Méthodes helper
    
    def _add_finding(self, id: str, title: str, description: str, 
                    severity: SeverityLevel, category: str, resource: Dict[str, Any],
                    rule_id: str, compliance: List[ComplianceStandard],
                    remediation: str, evidence: Optional[Dict[str, Any]] = None,
                    cve_references: Optional[List[str]] = None,
                    cvss_score: Optional[float] = None) -> None:
        """Ajoute un résultat de sécurité."""
        finding = SecurityFinding(
            id=f"{id}-{len(self.findings)}",
            title=title,
            description=description,
            severity=severity,
            category=category,
            resource_type=resource.get("kind", "Unknown"),
            resource_name=resource.get("metadata", {}).get("name", "Unknown"),
            namespace=resource.get("metadata", {}).get("namespace", self.namespace),
            rule_id=rule_id,
            compliance_standards=compliance,
            remediation=remediation,
            evidence=evidence,
            cve_references=cve_references,
            cvss_score=cvss_score
        )
        
        self.findings.append(finding)
    
    def _load_local_configurations(self) -> List[Dict[str, Any]]:
        """Charge les configurations depuis des fichiers locaux."""
        resources = []
        
        if not self.config_dir or not self.config_dir.exists():
            return resources
        
        # Recherche des fichiers YAML/JSON
        patterns = ["*.yaml", "*.yml", "*.json"]
        
        for pattern in patterns:
            for config_file in self.config_dir.rglob(pattern):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        if config_file.suffix == '.json':
                            content = json.load(f)
                        else:
                            content = yaml.safe_load_all(f)
                        
                        if isinstance(content, list):
                            resources.extend(content)
                        elif isinstance(content, dict):
                            resources.append(content)
                        else:
                            # Generator from yaml.safe_load_all
                            for doc in content:
                                if doc:
                                    resources.append(doc)
                                    
                except Exception as e:
                    print(f"⚠️ Erreur lors de la lecture de {config_file}: {e}")
        
        return resources
    
    def _load_cluster_configurations(self) -> List[Dict[str, Any]]:
        """Charge les configurations depuis le cluster."""
        resources = []
        
        resource_types = [
            "pods", "deployments", "services", "configmaps", "secrets",
            "ingresses", "networkpolicies", "roles", "clusterroles",
            "rolebindings", "clusterrolebindings", "serviceaccounts"
        ]
        
        for resource_type in resource_types:
            try:
                cmd = ["kubectl", "get", resource_type, "-n", self.namespace, "-o", "json"]
                if self.kubeconfig:
                    cmd.extend(["--kubeconfig", self.kubeconfig])
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                resources.extend(data.get("items", []))
                
            except subprocess.CalledProcessError:
                # Ressource non disponible dans ce namespace
                continue
        
        return resources
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """Charge les règles de sécurité."""
        # Base de règles de sécurité (peut être externalisée)
        return {
            "pod_security": {
                "runAsNonRoot": True,
                "readOnlyRootFilesystem": True,
                "privileged": False,
                "dangerous_capabilities": ["SYS_ADMIN", "NET_ADMIN", "SYS_TIME"]
            },
            "rbac_security": {
                "wildcard_permissions": False,
                "secrets_access": "limited"
            },
            "network_security": {
                "default_deny": True,
                "ingress_restrictions": True
            }
        }
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Charge les règles de conformité."""
        return {
            ComplianceStandard.GDPR.value: {
                "data_retention": True,
                "data_encryption": True,
                "access_logs": True
            },
            ComplianceStandard.CIS_KUBERNETES.value: {
                "pod_security_standards": True,
                "rbac_restrictions": True,
                "network_policies": True
            },
            ComplianceStandard.SOC2.value: {
                "audit_logging": True,
                "access_controls": True,
                "monitoring": True
            }
        }
    
    def _load_sensitive_patterns(self) -> List[str]:
        """Charge les patterns de données sensibles."""
        return [
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Numéros de carte de crédit
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN américain
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # Adresses IP
        ]
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Charge la base de données de vulnérabilités."""
        # Base simplifiée - en production, utiliser une vraie DB de vulnérabilités
        return {
            "kubernetes": {
                "1.20.0": ["CVE-2021-25735", "CVE-2021-25737"],
                "1.19.0": ["CVE-2020-8559", "CVE-2020-8558"]
            }
        }
    
    def _is_secret_encrypted_at_rest(self) -> bool:
        """Vérifie si les secrets sont chiffrés au repos."""
        # Simulation - en production, vérifier la configuration etcd
        return True
    
    def _contains_sensitive_patterns(self, content: str) -> bool:
        """Vérifie si le contenu contient des patterns sensibles."""
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _is_strong_password(self, password: str) -> bool:
        """Vérifie la force d'un mot de passe."""
        if len(password) < 12:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _calculate_compliance_scores(self) -> Dict[str, float]:
        """Calcule les scores de conformité."""
        scores = {}
        
        for standard in ComplianceStandard:
            standard_findings = [f for f in self.findings if standard in f.compliance_standards]
            
            if not standard_findings:
                scores[standard.value] = 100.0
            else:
                # Calcul basé sur la sévérité
                penalty = 0
                for finding in standard_findings:
                    if finding.severity == SeverityLevel.CRITICAL:
                        penalty += 25
                    elif finding.severity == SeverityLevel.HIGH:
                        penalty += 15
                    elif finding.severity == SeverityLevel.MEDIUM:
                        penalty += 10
                    elif finding.severity == SeverityLevel.LOW:
                        penalty += 5
                
                scores[standard.value] = max(0, 100 - penalty)
        
        return scores
    
    def _calculate_risk_score(self) -> float:
        """Calcule le score de risque global."""
        if not self.findings:
            return 0.0
        
        risk_score = 0
        for finding in self.findings:
            if finding.severity == SeverityLevel.CRITICAL:
                risk_score += 10
            elif finding.severity == SeverityLevel.HIGH:
                risk_score += 7
            elif finding.severity == SeverityLevel.MEDIUM:
                risk_score += 5
            elif finding.severity == SeverityLevel.LOW:
                risk_score += 2
            elif finding.severity == SeverityLevel.INFO:
                risk_score += 1
        
        return min(100, risk_score)
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur les résultats."""
        recommendations = []
        
        severity_counts = {}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        if severity_counts.get(SeverityLevel.CRITICAL, 0) > 0:
            recommendations.append("Corriger immédiatement tous les problèmes critiques")
        
        if severity_counts.get(SeverityLevel.HIGH, 0) > 5:
            recommendations.append("Prioriser la correction des problèmes de sévérité haute")
        
        if any("pod_security" in f.category for f in self.findings):
            recommendations.append("Implémenter des Pod Security Standards")
        
        if any("rbac" in f.category for f in self.findings):
            recommendations.append("Réviser les politiques RBAC pour le principe du moindre privilège")
        
        if any("secret" in f.category for f in self.findings):
            recommendations.append("Mettre en place un gestionnaire de secrets externe")
        
        return recommendations
    
    def export_report(self, format_type: str = "json") -> str:
        """Exporte le rapport dans différents formats."""
        if not hasattr(self, 'last_report'):
            raise ValueError("Aucun rapport disponible. Exécutez d'abord un scan.")
        
        if format_type == "json":
            return self._export_json_report()
        elif format_type == "sarif":
            return self._export_sarif_report()
        elif format_type == "html":
            return self._export_html_report()
        elif format_type == "csv":
            return self._export_csv_report()
        else:
            raise ValueError(f"Format non supporté: {format_type}")
    
    def _export_json_report(self) -> str:
        """Exporte le rapport au format JSON."""
        return json.dumps(asdict(self.last_report), indent=2, default=str)
    
    def _export_sarif_report(self) -> str:
        """Exporte le rapport au format SARIF."""
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "Spotify AI Agent Security Scanner",
                        "version": "2.0.0"
                    }
                },
                "results": []
            }]
        }
        
        for finding in self.last_report.findings:
            result = {
                "ruleId": finding.rule_id,
                "message": {"text": finding.description},
                "level": self._severity_to_sarif_level(finding.severity),
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": finding.file_path or f"{finding.resource_type}/{finding.resource_name}"
                        }
                    }
                }]
            }
            
            if finding.line_number:
                result["locations"][0]["physicalLocation"]["region"] = {
                    "startLine": finding.line_number
                }
            
            sarif_report["runs"][0]["results"].append(result)
        
        return json.dumps(sarif_report, indent=2)
    
    def _severity_to_sarif_level(self, severity: SeverityLevel) -> str:
        """Convertit la sévérité en niveau SARIF."""
        mapping = {
            SeverityLevel.CRITICAL: "error",
            SeverityLevel.HIGH: "error", 
            SeverityLevel.MEDIUM: "warning",
            SeverityLevel.LOW: "note",
            SeverityLevel.INFO: "note"
        }
        return mapping.get(severity, "note")
    
    def _export_html_report(self) -> str:
        """Exporte le rapport au format HTML."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .finding { margin: 10px 0; padding: 15px; border-left: 4px solid; }
        .critical { border-color: #d32f2f; background-color: #ffebee; }
        .high { border-color: #f57c00; background-color: #fff3e0; }
        .medium { border-color: #fbc02d; background-color: #fffde7; }
        .low { border-color: #388e3c; background-color: #e8f5e8; }
        .info { border-color: #1976d2; background-color: #e3f2fd; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p>Scan ID: {scan_id}</p>
        <p>Timestamp: {timestamp}</p>
        <p>Namespace: {namespace}</p>
        <p>Risk Score: {risk_score}/100</p>
        <p>Total Findings: {total_findings}</p>
    </div>
    
    <h2>Findings</h2>
    {findings_html}
    
    <h2>Recommendations</h2>
    <ul>
        {recommendations_html}
    </ul>
</body>
</html>
        """
        
        findings_html = ""
        for finding in self.last_report.findings:
            findings_html += f"""
            <div class="finding {finding.severity.value}">
                <h3>{finding.title}</h3>
                <p><strong>Severity:</strong> {finding.severity.value.upper()}</p>
                <p><strong>Resource:</strong> {finding.resource_type}/{finding.resource_name}</p>
                <p><strong>Description:</strong> {finding.description}</p>
                <p><strong>Remediation:</strong> {finding.remediation}</p>
            </div>
            """
        
        recommendations_html = ""
        for rec in self.last_report.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        
        return html_template.format(
            scan_id=self.last_report.scan_id,
            timestamp=self.last_report.timestamp,
            namespace=self.last_report.namespace,
            risk_score=self.last_report.risk_score,
            total_findings=len(self.last_report.findings),
            findings_html=findings_html,
            recommendations_html=recommendations_html
        )
    
    def _export_csv_report(self) -> str:
        """Exporte le rapport au format CSV."""
        lines = ["ID,Title,Severity,Category,Resource,Description,Remediation"]
        
        for finding in self.last_report.findings:
            line = f"{finding.id},{finding.title},{finding.severity.value},{finding.category},{finding.resource_type}/{finding.resource_name},\"{finding.description}\",\"{finding.remediation}\""
            lines.append(line)
        
        return "\n".join(lines)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Scanner de sécurité pour configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python security_scanner.py --namespace spotify-ai-agent-dev --full-scan
  python security_scanner.py --config-dir ./configs/ --compliance-check
  python security_scanner.py --export-report --format sarif --output security-report.sarif
        """
    )
    
    parser.add_argument(
        "--namespace", "-n",
        default="spotify-ai-agent-dev",
        help="Namespace Kubernetes à scanner"
    )
    
    parser.add_argument(
        "--kubeconfig", "-k",
        help="Chemin vers le fichier kubeconfig"
    )
    
    parser.add_argument(
        "--config-dir", "-d",
        type=Path,
        help="Répertoire des configurations locales à scanner"
    )
    
    # Types de scan
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan complet (tous les types)"
    )
    
    parser.add_argument(
        "--scan-types",
        nargs="+",
        choices=["configuration", "secrets", "rbac", "network", "compliance", "vulnerabilities", "best_practices"],
        help="Types de scan à exécuter"
    )
    
    parser.add_argument(
        "--compliance-check",
        action="store_true",
        help="Vérifications de conformité uniquement"
    )
    
    # Export et reporting
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Exporte le rapport de scan"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "sarif", "html", "csv"],
        default="json",
        help="Format du rapport"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Fichier de sortie pour le rapport"
    )
    
    # Filtres
    parser.add_argument(
        "--severity",
        choices=["critical", "high", "medium", "low", "info"],
        help="Filtrer par niveau de sévérité minimum"
    )
    
    parser.add_argument(
        "--category",
        help="Filtrer par catégorie"
    )
    
    args = parser.parse_args()
    
    try:
        # Création du scanner
        scanner = SecurityScanner(
            namespace=args.namespace,
            kubeconfig=args.kubeconfig,
            config_dir=args.config_dir
        )
        
        # Détermination des types de scan
        scan_types = None
        if args.full_scan:
            scan_types = ["configuration", "secrets", "rbac", "network", "compliance", "vulnerabilities", "best_practices"]
        elif args.compliance_check:
            scan_types = ["compliance"]
        elif args.scan_types:
            scan_types = args.scan_types
        
        # Exécution du scan
        report = scanner.scan_configurations(scan_types)
        scanner.last_report = report
        
        # Filtrage des résultats
        filtered_findings = report.findings
        
        if args.severity:
            min_severity = SeverityLevel(args.severity)
            severity_order = [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.INFO]
            min_index = severity_order.index(min_severity)
            filtered_findings = [f for f in filtered_findings if severity_order.index(f.severity) <= min_index]
        
        if args.category:
            filtered_findings = [f for f in filtered_findings if args.category.lower() in f.category.lower()]
        
        # Affichage des résultats
        print(f"\n📊 Résumé du scan:")
        print(f"   Ressources scannées: {report.total_resources}")
        print(f"   Problèmes détectés: {len(report.findings)}")
        print(f"   Score de risque: {report.risk_score:.1f}/100")
        print(f"   Durée du scan: {report.scan_duration:.1f}s")
        
        # Répartition par sévérité
        severity_counts = {}
        for finding in filtered_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        if severity_counts:
            print(f"\n🚨 Répartition par sévérité:")
            for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.INFO]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f"   {severity.value.upper()}: {count}")
        
        # Scores de conformité
        if report.compliance_scores:
            print(f"\n📋 Scores de conformité:")
            for standard, score in report.compliance_scores.items():
                print(f"   {standard.upper()}: {score:.1f}%")
        
        # Export du rapport
        if args.export_report:
            report_content = scanner.export_report(args.format)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"\n📄 Rapport exporté: {args.output}")
            else:
                print(f"\n📄 Rapport ({args.format}):")
                print(report_content)
        
        # Code de sortie basé sur les résultats
        if severity_counts.get(SeverityLevel.CRITICAL, 0) > 0:
            sys.exit(2)  # Problèmes critiques
        elif severity_counts.get(SeverityLevel.HIGH, 0) > 0:
            sys.exit(1)  # Problèmes importants
        else:
            sys.exit(0)  # Succès
    
    except Exception as e:
        print(f"❌ Erreur lors du scan: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
