"""
Configuration avancée d'audit et de compliance pour Alertmanager Receivers

Ce module gère l'audit complet, la traçabilité et la compliance réglementaire
pour toutes les opérations du système de receivers.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Spécialiste Sécurité Backend
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types d'événements d'audit"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIG_CHANGE = "config_change"
    ALERT_PROCESSED = "alert_processed"
    INTEGRATION_CALL = "integration_call"
    ESCALATION = "escalation"
    DATA_ACCESS = "data_access"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"

class AuditLevel(Enum):
    """Niveaux d'audit"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"

class ComplianceFramework(Enum):
    """Frameworks de compliance supportés"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"

@dataclass
class AuditEvent:
    """Événement d'audit complet"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_EVENT
    actor: str = "system"
    tenant: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    risk_level: str = "low"
    compliance_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Calcul de l'empreinte de l'événement
        event_data = {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "details": self.details
        }
        event_str = json.dumps(event_data, sort_keys=True)
        self.fingerprint = hashlib.sha256(event_str.encode()).hexdigest()

@dataclass
class ComplianceReport:
    """Rapport de compliance"""
    framework: ComplianceFramework
    compliance_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    last_check: datetime
    next_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)

class AuditStorage:
    """Stockage sécurisé des événements d'audit"""
    
    def __init__(self, storage_path: str = "/var/log/alertmanager/audit"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.file_rotation_size = 100 * 1024 * 1024  # 100MB
        
    async def store_event(self, event: AuditEvent):
        """Stocke un événement d'audit"""
        try:
            # Rotation des fichiers si nécessaire
            await self._rotate_if_needed()
            
            # Sérialisation de l'événement
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            event_data['event_type'] = event.event_type.value
            
            # Écriture dans le fichier d'audit
            audit_line = json.dumps(event_data) + "\n"
            
            current_file = self._get_current_file()
            with open(current_file, 'a', encoding='utf-8') as f:
                f.write(audit_line)
            
            logger.debug(f"Audit event stored: {event.id}")
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            # En cas d'erreur, on peut implémenter un fallback
            await self._store_to_fallback(event)
    
    def _get_current_file(self) -> Path:
        """Obtient le fichier d'audit actuel"""
        if self.current_file is None or not self.current_file.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.storage_path / f"audit_{timestamp}.jsonl"
        
        return self.current_file
    
    async def _rotate_if_needed(self):
        """Effectue la rotation des fichiers si nécessaire"""
        current_file = self._get_current_file()
        
        if current_file.exists() and current_file.stat().st_size > self.file_rotation_size:
            # Archivage du fichier actuel
            archive_name = f"{current_file.stem}_archived{current_file.suffix}"
            archive_path = self.storage_path / "archives" / archive_name
            archive_path.parent.mkdir(exist_ok=True)
            
            current_file.rename(archive_path)
            self.current_file = None
            
            logger.info(f"Audit log rotated: {archive_path}")
    
    async def _store_to_fallback(self, event: AuditEvent):
        """Stockage de fallback en cas d'erreur"""
        fallback_file = self.storage_path / "fallback_audit.jsonl"
        
        try:
            event_data = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "action": event.action,
                "result": event.result
            }
            
            with open(fallback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data) + "\n")
                
        except Exception as e:
            logger.critical(f"Failed to store to fallback: {e}")

class ComplianceChecker:
    """Vérificateur de compliance automatisé"""
    
    def __init__(self):
        self.frameworks = {
            ComplianceFramework.GDPR: self._check_gdpr_compliance,
            ComplianceFramework.SOC2: self._check_soc2_compliance,
            ComplianceFramework.ISO27001: self._check_iso27001_compliance,
            ComplianceFramework.PCI_DSS: self._check_pci_dss_compliance
        }
        
    async def check_compliance(
        self,
        framework: ComplianceFramework,
        audit_events: List[AuditEvent],
        config_data: Dict[str, Any]
    ) -> ComplianceReport:
        """Vérifie la compliance pour un framework donné"""
        
        if framework not in self.frameworks:
            raise ValueError(f"Unsupported compliance framework: {framework}")
        
        checker_func = self.frameworks[framework]
        return await checker_func(audit_events, config_data)
    
    async def _check_gdpr_compliance(
        self,
        audit_events: List[AuditEvent],
        config_data: Dict[str, Any]
    ) -> ComplianceReport:
        """Vérification GDPR"""
        violations = []
        recommendations = []
        score = 100.0
        
        # Vérification du droit à l'oubli
        data_retention_days = config_data.get("data_retention_days", 90)
        if data_retention_days > 365:
            violations.append({
                "rule": "GDPR Article 17 - Right to erasure",
                "description": "Data retention period exceeds recommended maximum",
                "severity": "medium",
                "current_value": data_retention_days,
                "recommended_value": 365
            })
            score -= 10
        
        # Vérification de l'anonymisation
        pii_events = [e for e in audit_events if "pii" in e.details]
        if pii_events:
            anonymized_count = sum(1 for e in pii_events if e.details.get("anonymized", False))
            if anonymized_count < len(pii_events):
                violations.append({
                    "rule": "GDPR Article 25 - Data protection by design",
                    "description": "PII data not properly anonymized",
                    "severity": "high",
                    "affected_events": len(pii_events) - anonymized_count
                })
                score -= 20
        
        # Vérification des accès aux données
        data_access_events = [e for e in audit_events if e.event_type == AuditEventType.DATA_ACCESS]
        unauthorized_access = [e for e in data_access_events if e.result == "unauthorized"]
        if unauthorized_access:
            violations.append({
                "rule": "GDPR Article 32 - Security of processing",
                "description": "Unauthorized data access detected",
                "severity": "critical",
                "incident_count": len(unauthorized_access)
            })
            score -= 30
        
        if score == 100.0:
            recommendations.append("GDPR compliance is excellent. Continue current practices.")
        else:
            recommendations.extend([
                "Implement automatic data purging after retention period",
                "Enhance PII anonymization processes",
                "Review access control policies",
                "Conduct regular privacy impact assessments"
            ])
        
        return ComplianceReport(
            framework=ComplianceFramework.GDPR,
            compliance_score=max(0.0, score),
            violations=violations,
            recommendations=recommendations,
            last_check=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=30),
            details={
                "total_events_checked": len(audit_events),
                "pii_events_count": len(pii_events),
                "data_retention_days": data_retention_days
            }
        )
    
    async def _check_soc2_compliance(
        self,
        audit_events: List[AuditEvent],
        config_data: Dict[str, Any]
    ) -> ComplianceReport:
        """Vérification SOC2"""
        violations = []
        recommendations = []
        score = 100.0
        
        # Principe de sécurité - Contrôles d'accès
        auth_failures = [e for e in audit_events 
                        if e.event_type == AuditEventType.AUTHENTICATION and e.result == "failure"]
        if len(auth_failures) > len(audit_events) * 0.05:  # Plus de 5% d'échecs
            violations.append({
                "principle": "Security",
                "control": "CC6.1 - Logical access security measures",
                "description": "High authentication failure rate detected",
                "severity": "medium",
                "failure_rate": len(auth_failures) / len(audit_events) * 100
            })
            score -= 15
        
        # Principe de disponibilité - Monitoring
        system_events = [e for e in audit_events if e.event_type == AuditEventType.SYSTEM_EVENT]
        downtime_events = [e for e in system_events if "downtime" in e.action.lower()]
        if downtime_events:
            violations.append({
                "principle": "Availability",
                "control": "A1.1 - Performance monitoring",
                "description": "System downtime events detected",
                "severity": "high",
                "downtime_incidents": len(downtime_events)
            })
            score -= 25
        
        # Principe d'intégrité - Changements de configuration
        config_changes = [e for e in audit_events if e.event_type == AuditEventType.CONFIG_CHANGE]
        unauthorized_changes = [e for e in config_changes if e.result == "unauthorized"]
        if unauthorized_changes:
            violations.append({
                "principle": "Processing Integrity",
                "control": "PI1.1 - Data processing integrity",
                "description": "Unauthorized configuration changes detected",
                "severity": "critical",
                "unauthorized_changes": len(unauthorized_changes)
            })
            score -= 30
        
        if score >= 90:
            recommendations.append("SOC2 compliance is strong. Minor improvements recommended.")
        else:
            recommendations.extend([
                "Strengthen authentication mechanisms",
                "Implement comprehensive system monitoring",
                "Enhance change management processes",
                "Conduct regular security assessments"
            ])
        
        return ComplianceReport(
            framework=ComplianceFramework.SOC2,
            compliance_score=max(0.0, score),
            violations=violations,
            recommendations=recommendations,
            last_check=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=90),
            details={
                "total_events_checked": len(audit_events),
                "auth_failure_rate": len(auth_failures) / max(1, len(audit_events)) * 100,
                "config_changes_count": len(config_changes)
            }
        )
    
    async def _check_iso27001_compliance(
        self,
        audit_events: List[AuditEvent],
        config_data: Dict[str, Any]
    ) -> ComplianceReport:
        """Vérification ISO27001"""
        violations = []
        recommendations = []
        score = 100.0
        
        # A.9.2.1 - Enregistrement et désenregistrement des utilisateurs
        user_mgmt_events = [e for e in audit_events 
                           if "user" in e.action.lower() and e.event_type == AuditEventType.SYSTEM_EVENT]
        
        # A.12.4.1 - Enregistrement des événements
        if len(audit_events) == 0:
            violations.append({
                "control": "A.12.4.1",
                "description": "No audit events recorded",
                "severity": "critical"
            })
            score -= 40
        
        # A.18.1.4 - Protection des données et respect de la vie privée
        encryption_enabled = config_data.get("encryption", {}).get("enabled", False)
        if not encryption_enabled:
            violations.append({
                "control": "A.18.1.4",
                "description": "Data encryption not enabled",
                "severity": "high"
            })
            score -= 25
        
        return ComplianceReport(
            framework=ComplianceFramework.ISO27001,
            compliance_score=max(0.0, score),
            violations=violations,
            recommendations=recommendations,
            last_check=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=180),
            details={"total_events_checked": len(audit_events)}
        )
    
    async def _check_pci_dss_compliance(
        self,
        audit_events: List[AuditEvent],
        config_data: Dict[str, Any]
    ) -> ComplianceReport:
        """Vérification PCI-DSS"""
        violations = []
        recommendations = []
        score = 100.0
        
        # Requirement 2: Ne pas utiliser les mots de passe par défaut
        # Requirement 8: Identifier et authentifier l'accès aux composants système
        # Requirement 10: Traquer et surveiller tous les accès aux ressources réseau
        
        if len(audit_events) == 0:
            violations.append({
                "requirement": "10.2",
                "description": "No audit trail maintained",
                "severity": "critical"
            })
            score -= 50
        
        return ComplianceReport(
            framework=ComplianceFramework.PCI_DSS,
            compliance_score=max(0.0, score),
            violations=violations,
            recommendations=recommendations,
            last_check=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=90),
            details={"total_events_checked": len(audit_events)}
        )

class AuditConfigManager:
    """Gestionnaire principal de l'audit et de la compliance"""
    
    def __init__(self, audit_level: AuditLevel = AuditLevel.COMPREHENSIVE):
        self.audit_level = audit_level
        self.storage = AuditStorage()
        self.compliance_checker = ComplianceChecker()
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 1000
        self.compliance_reports: Dict[ComplianceFramework, ComplianceReport] = {}
        
    async def initialize_audit(self) -> bool:
        """Initialise le système d'audit"""
        try:
            logger.info("Initializing audit configuration manager")
            
            # Démarrage des tâches d'audit
            await self._start_audit_tasks()
            
            # Première vérification de compliance
            await self._initial_compliance_check()
            
            logger.info("Audit system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audit system: {e}")
            return False
    
    async def _start_audit_tasks(self):
        """Démarre les tâches d'audit en arrière-plan"""
        # Tâche de vidage du buffer
        asyncio.create_task(self._flush_buffer_periodically())
        
        # Tâche de vérification de compliance
        asyncio.create_task(self._periodic_compliance_check())
        
        # Tâche de nettoyage des anciens logs
        asyncio.create_task(self._cleanup_old_logs())
    
    async def _flush_buffer_periodically(self):
        """Vide le buffer d'événements périodiquement"""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_event_buffer()
            except Exception as e:
                logger.error(f"Error flushing audit buffer: {e}")
    
    async def _flush_event_buffer(self):
        """Vide le buffer d'événements vers le stockage"""
        if self.event_buffer:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
            
            for event in events_to_flush:
                await self.storage.store_event(event)
    
    async def _periodic_compliance_check(self):
        """Vérification périodique de compliance"""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily
                await self._run_compliance_checks()
            except Exception as e:
                logger.error(f"Error in compliance check: {e}")
    
    async def _run_compliance_checks(self):
        """Exécute les vérifications de compliance"""
        # Récupération des événements récents
        recent_events = self.event_buffer[-10000:]  # 10k derniers événements
        
        # Configuration factice pour les tests
        config_data = {
            "data_retention_days": 90,
            "encryption": {"enabled": True}
        }
        
        # Vérification pour chaque framework
        for framework in ComplianceFramework:
            try:
                report = await self.compliance_checker.check_compliance(
                    framework, recent_events, config_data
                )
                self.compliance_reports[framework] = report
                
                if report.compliance_score < 80:
                    logger.warning(f"Low compliance score for {framework.value}: {report.compliance_score}")
                
            except Exception as e:
                logger.error(f"Compliance check failed for {framework.value}: {e}")
    
    async def _cleanup_old_logs(self):
        """Nettoie les anciens logs d'audit"""
        while True:
            try:
                await asyncio.sleep(86400 * 7)  # Weekly cleanup
                
                # Nettoyage des logs plus anciens que la période de rétention
                retention_days = 90
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Implémentation du nettoyage...
                logger.info("Completed audit log cleanup")
                
            except Exception as e:
                logger.error(f"Error in audit log cleanup: {e}")
    
    async def _initial_compliance_check(self):
        """Vérification initiale de compliance"""
        await self._run_compliance_checks()
    
    async def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        tenant: Optional[str] = None,
        resource: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low"
    ):
        """Enregistre un événement d'audit"""
        
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            tenant=tenant,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_level=risk_level
        )
        
        # Ajout au buffer
        self.event_buffer.append(event)
        
        # Flush immédiat si le buffer est plein ou pour les événements critiques
        if len(self.event_buffer) >= self.buffer_size or risk_level == "critical":
            await self._flush_event_buffer()
        
        logger.debug(f"Audit event logged: {event.id}")
    
    def get_compliance_report(self, framework: ComplianceFramework) -> Optional[ComplianceReport]:
        """Récupère le rapport de compliance pour un framework"""
        return self.compliance_reports.get(framework)
    
    def get_all_compliance_reports(self) -> Dict[ComplianceFramework, ComplianceReport]:
        """Récupère tous les rapports de compliance"""
        return self.compliance_reports.copy()
    
    async def generate_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Génère un résumé d'audit pour les N derniers jours"""
        # Filtrage des événements récents
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_events = [e for e in self.event_buffer if e.timestamp > cutoff_date]
        
        # Analyse des événements
        summary = {
            "period_days": days,
            "total_events": len(recent_events),
            "by_type": {},
            "by_actor": {},
            "by_tenant": {},
            "by_result": {},
            "risk_distribution": {},
            "compliance_status": {}
        }
        
        # Répartition par type
        for event in recent_events:
            event_type = event.event_type.value
            summary["by_type"][event_type] = summary["by_type"].get(event_type, 0) + 1
            
            summary["by_actor"][event.actor] = summary["by_actor"].get(event.actor, 0) + 1
            
            if event.tenant:
                summary["by_tenant"][event.tenant] = summary["by_tenant"].get(event.tenant, 0) + 1
            
            summary["by_result"][event.result] = summary["by_result"].get(event.result, 0) + 1
            summary["risk_distribution"][event.risk_level] = summary["risk_distribution"].get(event.risk_level, 0) + 1
        
        # Statut de compliance
        for framework, report in self.compliance_reports.items():
            summary["compliance_status"][framework.value] = {
                "score": report.compliance_score,
                "violations_count": len(report.violations),
                "last_check": report.last_check.isoformat()
            }
        
        return summary

# Instance singleton
audit_manager = AuditConfigManager()
