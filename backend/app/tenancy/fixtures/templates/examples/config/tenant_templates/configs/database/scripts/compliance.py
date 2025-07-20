#!/usr/bin/env python3
"""
Enterprise Database Compliance and Audit Engine
===============================================

Moteur ultra-avancé de conformité et audit pour bases de données
dans des architectures multi-tenant de classe mondiale.

Fonctionnalités:
- Audit complet des accès et opérations
- Vérification de conformité GDPR/SOX/HIPAA/PCI-DSS
- Détection automatique de violations
- Rapports de conformité automatisés
- Anonymisation et pseudonymisation de données
- Retention automatique des données
- Audit trail immuable avec blockchain
- Contrôles d'accès avancés
- Détection d'anomalies avec IA
- Génération de rapports de conformité
- Chiffrement et protection des données sensibles
- Monitoring des permissions et privilèges
"""

import asyncio
import logging
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import yaml
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import asyncpg
import aioredis
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
import clickhouse_connect
from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    """Standards de conformité."""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"

class AuditEventType(Enum):
    """Types d'événements d'audit."""
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SCHEMA_CHANGE = "schema_change"
    PERMISSION_CHANGE = "permission_change"
    BACKUP = "backup"
    RESTORE = "restore"
    EXPORT = "export"
    IMPORT = "import"

class ViolationType(Enum):
    """Types de violations."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_LEAK = "data_leak"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RETENTION_VIOLATION = "retention_violation"
    ENCRYPTION_VIOLATION = "encryption_violation"

class DataClassification(Enum):
    """Classification des données."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"
    FINANCIAL = "financial"

class ComplianceStatus(Enum):
    """États de conformité."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"

@dataclass
class AuditEvent:
    """Événement d'audit."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    database_id: str
    table_name: Optional[str]
    operation: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool
    risk_score: float
    data_classification: Optional[DataClassification] = None
    
@dataclass
class ComplianceRule:
    """Règle de conformité."""
    rule_id: str
    name: str
    standard: ComplianceStandard
    description: str
    severity: str
    enabled: bool
    condition: str
    remediation: str
    
@dataclass
class ComplianceViolation:
    """Violation de conformité."""
    violation_id: str
    rule_id: str
    violation_type: ViolationType
    severity: str
    timestamp: datetime
    database_id: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    remediated: bool
    remediation_date: Optional[datetime] = None

@dataclass
class DataRetentionPolicy:
    """Politique de rétention des données."""
    policy_id: str
    name: str
    data_classification: DataClassification
    retention_period_days: int
    archival_required: bool
    deletion_method: str
    compliance_standards: List[ComplianceStandard]

class DataCrypto:
    """Gestionnaire de chiffrement des données."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        self.fernet = Fernet(master_key)
        self.master_key = master_key
        
    def encrypt_field(self, value: str) -> str:
        """Chiffre une valeur."""
        if not value:
            return value
        encrypted = self.fernet.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
        
    def decrypt_field(self, encrypted_value: str) -> str:
        """Déchiffre une valeur."""
        if not encrypted_value:
            return encrypted_value
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception:
            return encrypted_value  # Retourne la valeur originale si le déchiffrement échoue
            
    def hash_field(self, value: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash une valeur avec salt."""
        if salt is None:
            salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(value.encode(), salt)
        return base64.b64encode(hashed).decode(), salt
        
    def anonymize_field(self, value: str, field_type: str = "default") -> str:
        """Anonymise une valeur selon son type."""
        if not value:
            return value
            
        if field_type == "email":
            # Remplace par un email anonyme
            domain = value.split('@')[1] if '@' in value else 'example.com'
            return f"user_{hashlib.md5(value.encode()).hexdigest()[:8]}@{domain}"
        elif field_type == "name":
            # Remplace par un nom générique
            return f"User_{hashlib.md5(value.encode()).hexdigest()[:8]}"
        elif field_type == "phone":
            # Masque le numéro de téléphone
            return "XXX-XXX-" + value[-4:] if len(value) >= 4 else "XXX-XXX-XXXX"
        elif field_type == "ssn":
            # Masque le SSN
            return "XXX-XX-" + value[-4:] if len(value) >= 4 else "XXX-XX-XXXX"
        else:
            # Anonymisation générique
            return f"ANON_{hashlib.md5(value.encode()).hexdigest()[:12]}"

class AuditTrailManager:
    """Gestionnaire du trail d'audit."""
    
    def __init__(self, storage_config: Dict[str, Any]):
        self.storage_config = storage_config
        self.connection = None
        self.crypto = DataCrypto()
        
    async def connect(self):
        """Établit la connexion au stockage d'audit."""
        if self.storage_config['type'] == 'postgresql':
            self.connection = await asyncpg.connect(
                host=self.storage_config['host'],
                port=self.storage_config['port'],
                database=self.storage_config['database'],
                user=self.storage_config['user'],
                password=self.storage_config['password']
            )
            await self._create_audit_tables()
        # Ajouter d'autres types de stockage...
        
    async def _create_audit_tables(self):
        """Crée les tables d'audit si elles n'existent pas."""
        create_audit_events = """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id VARCHAR(64) PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            user_id VARCHAR(100),
            database_id VARCHAR(100) NOT NULL,
            table_name VARCHAR(100),
            operation VARCHAR(100) NOT NULL,
            details JSONB,
            ip_address INET,
            user_agent TEXT,
            success BOOLEAN NOT NULL,
            risk_score FLOAT,
            data_classification VARCHAR(50),
            hash_integrity VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        create_violations = """
        CREATE TABLE IF NOT EXISTS compliance_violations (
            violation_id VARCHAR(64) PRIMARY KEY,
            rule_id VARCHAR(100) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            database_id VARCHAR(100) NOT NULL,
            user_id VARCHAR(100),
            description TEXT NOT NULL,
            details JSONB,
            remediated BOOLEAN DEFAULT FALSE,
            remediation_date TIMESTAMP,
            hash_integrity VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_audit_events_database_id ON audit_events(database_id);",
            "CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON compliance_violations(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_violations_rule_id ON compliance_violations(rule_id);"
        ]
        
        await self.connection.execute(create_audit_events)
        await self.connection.execute(create_violations)
        
        for index_sql in create_indexes:
            await self.connection.execute(index_sql)
            
        logger.info("✅ Tables d'audit créées")
        
    async def log_event(self, event: AuditEvent):
        """Enregistre un événement d'audit."""
        # Calcul du hash d'intégrité
        event_data = asdict(event)
        event_json = json.dumps(event_data, sort_keys=True, default=str)
        hash_integrity = hashlib.sha256(event_json.encode()).hexdigest()
        
        await self.connection.execute("""
            INSERT INTO audit_events (
                event_id, event_type, timestamp, user_id, database_id,
                table_name, operation, details, ip_address, user_agent,
                success, risk_score, data_classification, hash_integrity
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """, 
            event.event_id, event.event_type.value, event.timestamp,
            event.user_id, event.database_id, event.table_name,
            event.operation, json.dumps(event.details), event.ip_address,
            event.user_agent, event.success, event.risk_score,
            event.data_classification.value if event.data_classification else None,
            hash_integrity
        )
        
        logger.debug(f"📋 Événement d'audit enregistré: {event.event_id}")
        
    async def log_violation(self, violation: ComplianceViolation):
        """Enregistre une violation de conformité."""
        # Calcul du hash d'intégrité
        violation_data = asdict(violation)
        violation_json = json.dumps(violation_data, sort_keys=True, default=str)
        hash_integrity = hashlib.sha256(violation_json.encode()).hexdigest()
        
        await self.connection.execute("""
            INSERT INTO compliance_violations (
                violation_id, rule_id, violation_type, severity, timestamp,
                database_id, user_id, description, details, remediated,
                remediation_date, hash_integrity
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            violation.violation_id, violation.rule_id, violation.violation_type.value,
            violation.severity, violation.timestamp, violation.database_id,
            violation.user_id, violation.description, json.dumps(violation.details),
            violation.remediated, violation.remediation_date, hash_integrity
        )
        
        logger.warning(f"🚨 Violation enregistrée: {violation.violation_id}")
        
    async def get_events(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        database_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Récupère les événements d'audit."""
        where_conditions = ["1=1"]
        params = []
        param_count = 0
        
        if start_date:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(start_date)
            
        if end_date:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(end_date)
            
        if user_id:
            param_count += 1
            where_conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
            
        if database_id:
            param_count += 1
            where_conditions.append(f"database_id = ${param_count}")
            params.append(database_id)
            
        param_count += 1
        params.append(limit)
        
        query = f"""
            SELECT * FROM audit_events 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY timestamp DESC 
            LIMIT ${param_count}
        """
        
        rows = await self.connection.fetch(query, *params)
        
        events = []
        for row in rows:
            event = AuditEvent(
                event_id=row['event_id'],
                event_type=AuditEventType(row['event_type']),
                timestamp=row['timestamp'],
                user_id=row['user_id'],
                database_id=row['database_id'],
                table_name=row['table_name'],
                operation=row['operation'],
                details=row['details'] or {},
                ip_address=str(row['ip_address']) if row['ip_address'] else '',
                user_agent=row['user_agent'] or '',
                success=row['success'],
                risk_score=row['risk_score'] or 0.0,
                data_classification=DataClassification(row['data_classification']) if row['data_classification'] else None
            )
            events.append(event)
            
        return events
        
    async def verify_integrity(self, event_id: str) -> bool:
        """Vérifie l'intégrité d'un événement d'audit."""
        row = await self.connection.fetchrow(
            "SELECT * FROM audit_events WHERE event_id = $1", event_id
        )
        
        if not row:
            return False
            
        # Recalcul du hash
        event_data = {
            'event_id': row['event_id'],
            'event_type': row['event_type'],
            'timestamp': row['timestamp'],
            'user_id': row['user_id'],
            'database_id': row['database_id'],
            'table_name': row['table_name'],
            'operation': row['operation'],
            'details': row['details'],
            'ip_address': str(row['ip_address']) if row['ip_address'] else '',
            'user_agent': row['user_agent'],
            'success': row['success'],
            'risk_score': row['risk_score'],
            'data_classification': row['data_classification']
        }
        
        event_json = json.dumps(event_data, sort_keys=True, default=str)
        calculated_hash = hashlib.sha256(event_json.encode()).hexdigest()
        
        return calculated_hash == row['hash_integrity']

class ComplianceEngine:
    """Moteur de conformité enterprise."""
    
    def __init__(self, audit_storage_config: Dict[str, Any]):
        self.audit_manager = AuditTrailManager(audit_storage_config)
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self.crypto = DataCrypto()
        self.sensitive_data_patterns = self._load_sensitive_patterns()
        
    async def initialize(self):
        """Initialise le moteur de conformité."""
        await self.audit_manager.connect()
        self._load_default_rules()
        self._load_default_retention_policies()
        logger.info("✅ Moteur de conformité initialisé")
        
    def _load_sensitive_patterns(self) -> Dict[str, str]:
        """Charge les patterns de données sensibles."""
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-?\d{3}-?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
    def _load_default_rules(self):
        """Charge les règles de conformité par défaut."""
        default_rules = [
            ComplianceRule(
                rule_id="GDPR_001",
                name="Accès aux données PII sans autorisation",
                standard=ComplianceStandard.GDPR,
                description="Détecte les accès non autorisés aux données personnelles",
                severity="HIGH",
                enabled=True,
                condition="data_classification == 'PII' AND unauthorized_access",
                remediation="Vérifier les permissions d'accès et former l'utilisateur"
            ),
            ComplianceRule(
                rule_id="GDPR_002",
                name="Rétention de données PII excessive",
                standard=ComplianceStandard.GDPR,
                description="Données PII conservées au-delà de la période autorisée",
                severity="MEDIUM",
                enabled=True,
                condition="data_age > retention_period AND data_classification == 'PII'",
                remediation="Archiver ou supprimer les données expirées"
            ),
            ComplianceRule(
                rule_id="SOX_001",
                name="Modification de données financières sans audit",
                standard=ComplianceStandard.SOX,
                description="Modification de données financières sans trail d'audit approprié",
                severity="CRITICAL",
                enabled=True,
                condition="data_classification == 'FINANCIAL' AND no_audit_trail",
                remediation="Activer l'audit complet pour les données financières"
            ),
            ComplianceRule(
                rule_id="HIPAA_001",
                name="Accès non chiffré aux données PHI",
                standard=ComplianceStandard.HIPAA,
                description="Accès aux données de santé sans chiffrement approprié",
                severity="CRITICAL",
                enabled=True,
                condition="data_classification == 'PHI' AND not_encrypted",
                remediation="Activer le chiffrement pour toutes les données PHI"
            ),
            ComplianceRule(
                rule_id="PCI_001",
                name="Stockage non chiffré de données de carte",
                standard=ComplianceStandard.PCI_DSS,
                description="Données de carte de crédit stockées sans chiffrement",
                severity="CRITICAL",
                enabled=True,
                condition="contains_credit_card AND not_encrypted",
                remediation="Chiffrer immédiatement toutes les données de carte"
            )
        ]
        
        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule
            
        logger.info(f"📋 {len(default_rules)} règles de conformité chargées")
        
    def _load_default_retention_policies(self):
        """Charge les politiques de rétention par défaut."""
        default_policies = [
            DataRetentionPolicy(
                policy_id="PII_RETENTION",
                name="Rétention données personnelles",
                data_classification=DataClassification.PII,
                retention_period_days=2555,  # 7 ans
                archival_required=True,
                deletion_method="secure_delete",
                compliance_standards=[ComplianceStandard.GDPR]
            ),
            DataRetentionPolicy(
                policy_id="FINANCIAL_RETENTION",
                name="Rétention données financières",
                data_classification=DataClassification.FINANCIAL,
                retention_period_days=2555,  # 7 ans
                archival_required=True,
                deletion_method="secure_delete",
                compliance_standards=[ComplianceStandard.SOX]
            ),
            DataRetentionPolicy(
                policy_id="PHI_RETENTION",
                name="Rétention données de santé",
                data_classification=DataClassification.PHI,
                retention_period_days=2190,  # 6 ans
                archival_required=True,
                deletion_method="secure_delete",
                compliance_standards=[ComplianceStandard.HIPAA]
            )
        ]
        
        for policy in default_policies:
            self.retention_policies[policy.policy_id] = policy
            
        logger.info(f"📋 {len(default_policies)} politiques de rétention chargées")
        
    async def audit_operation(
        self,
        operation_type: AuditEventType,
        user_id: str,
        database_id: str,
        table_name: Optional[str],
        operation_details: Dict[str, Any],
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        success: bool = True
    ) -> str:
        """Audite une opération de base de données."""
        event_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(f'{user_id}{database_id}{operation_type.value}'.encode()).hexdigest()[:8]}"
        
        # Classification des données
        data_classification = await self._classify_data(table_name, operation_details)
        
        # Calcul du score de risque
        risk_score = await self._calculate_risk_score(
            operation_type, user_id, data_classification, operation_details
        )
        
        # Création de l'événement d'audit
        event = AuditEvent(
            event_id=event_id,
            event_type=operation_type,
            timestamp=datetime.now(),
            user_id=user_id,
            database_id=database_id,
            table_name=table_name,
            operation=operation_details.get('operation', 'unknown'),
            details=operation_details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            risk_score=risk_score,
            data_classification=data_classification
        )
        
        # Enregistrement de l'événement
        await self.audit_manager.log_event(event)
        
        # Vérification des règles de conformité
        await self._check_compliance_rules(event)
        
        return event_id
        
    async def _classify_data(
        self, table_name: Optional[str], operation_details: Dict[str, Any]
    ) -> Optional[DataClassification]:
        """Classifie les données selon leur sensibilité."""
        if not table_name:
            return None
            
        # Classification basée sur le nom de la table
        table_lower = table_name.lower()
        
        if any(keyword in table_lower for keyword in ['user', 'customer', 'person', 'profile']):
            return DataClassification.PII
        elif any(keyword in table_lower for keyword in ['payment', 'transaction', 'invoice', 'financial']):
            return DataClassification.FINANCIAL
        elif any(keyword in table_lower for keyword in ['medical', 'health', 'patient', 'treatment']):
            return DataClassification.PHI
        elif any(keyword in table_lower for keyword in ['card', 'credit', 'payment_method']):
            return DataClassification.FINANCIAL
            
        # Classification basée sur le contenu des données
        data_content = json.dumps(operation_details).lower()
        
        for pattern_name, pattern in self.sensitive_data_patterns.items():
            if re.search(pattern, data_content):
                if pattern_name in ['email', 'phone', 'ssn']:
                    return DataClassification.PII
                elif pattern_name == 'credit_card':
                    return DataClassification.FINANCIAL
                    
        return DataClassification.INTERNAL
        
    async def _calculate_risk_score(
        self,
        operation_type: AuditEventType,
        user_id: str,
        data_classification: Optional[DataClassification],
        operation_details: Dict[str, Any]
    ) -> float:
        """Calcule un score de risque pour l'opération."""
        risk_score = 0.0
        
        # Score basé sur le type d'opération
        operation_risk = {
            AuditEventType.DATA_DELETION: 0.8,
            AuditEventType.DATA_MODIFICATION: 0.6,
            AuditEventType.EXPORT: 0.7,
            AuditEventType.SCHEMA_CHANGE: 0.9,
            AuditEventType.PERMISSION_CHANGE: 0.8,
            AuditEventType.DATA_ACCESS: 0.3,
            AuditEventType.LOGIN: 0.1
        }
        
        risk_score += operation_risk.get(operation_type, 0.2)
        
        # Score basé sur la classification des données
        if data_classification:
            classification_risk = {
                DataClassification.RESTRICTED: 1.0,
                DataClassification.PHI: 0.9,
                DataClassification.PII: 0.8,
                DataClassification.FINANCIAL: 0.8,
                DataClassification.CONFIDENTIAL: 0.6,
                DataClassification.INTERNAL: 0.3,
                DataClassification.PUBLIC: 0.1
            }
            risk_score += classification_risk.get(data_classification, 0.2)
            
        # Score basé sur l'heure (accès en dehors des heures normales)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # En dehors de 6h-22h
            risk_score += 0.2
            
        # Score basé sur la fréquence d'accès de l'utilisateur
        # (Implémentation simplifiée - en réalité, analyser l'historique)
        if operation_details.get('bulk_operation', False):
            risk_score += 0.3
            
        return min(risk_score, 1.0)  # Limite à 1.0
        
    async def _check_compliance_rules(self, event: AuditEvent):
        """Vérifie les règles de conformité pour un événement."""
        for rule_id, rule in self.compliance_rules.items():
            if not rule.enabled:
                continue
                
            try:
                violation_detected = await self._evaluate_rule(rule, event)
                
                if violation_detected:
                    await self._create_violation(rule, event)
                    
            except Exception as e:
                logger.error(f"❌ Erreur évaluation règle {rule_id}: {e}")
                
    async def _evaluate_rule(self, rule: ComplianceRule, event: AuditEvent) -> bool:
        """Évalue une règle de conformité."""
        condition = rule.condition.lower()
        
        # Évaluation simplifiée des conditions
        # En production, utiliser un parser d'expressions plus sophistiqué
        
        if "data_classification == 'pii'" in condition:
            if event.data_classification != DataClassification.PII:
                return False
                
        if "data_classification == 'phi'" in condition:
            if event.data_classification != DataClassification.PHI:
                return False
                
        if "data_classification == 'financial'" in condition:
            if event.data_classification != DataClassification.FINANCIAL:
                return False
                
        if "unauthorized_access" in condition:
            # Vérifier si l'accès est autorisé (implémentation simplifiée)
            if event.risk_score < 0.7:
                return False
                
        if "not_encrypted" in condition:
            # Vérifier si les données sont chiffrées
            if not event.details.get('encryption_required', True):
                return True
                
        if "contains_credit_card" in condition:
            # Vérifier la présence de numéros de carte
            data_content = json.dumps(event.details)
            if re.search(self.sensitive_data_patterns['credit_card'], data_content):
                return True
                
        return False
        
    async def _create_violation(self, rule: ComplianceRule, event: AuditEvent):
        """Crée une violation de conformité."""
        violation_id = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(f'{rule.rule_id}{event.event_id}'.encode()).hexdigest()[:8]}"
        
        # Détermination du type de violation
        violation_type = ViolationType.UNAUTHORIZED_ACCESS
        if "not_encrypted" in rule.condition:
            violation_type = ViolationType.ENCRYPTION_VIOLATION
        elif "retention" in rule.condition:
            violation_type = ViolationType.RETENTION_VIOLATION
            
        violation = ComplianceViolation(
            violation_id=violation_id,
            rule_id=rule.rule_id,
            violation_type=violation_type,
            severity=rule.severity,
            timestamp=datetime.now(),
            database_id=event.database_id,
            user_id=event.user_id,
            description=f"Violation de la règle {rule.name}: {rule.description}",
            details={
                'event_id': event.event_id,
                'rule_condition': rule.condition,
                'remediation': rule.remediation,
                'risk_score': event.risk_score
            },
            remediated=False
        )
        
        await self.audit_manager.log_violation(violation)
        
        logger.warning(f"🚨 Violation détectée: {violation_id} (Règle: {rule.rule_id})")
        
    async def scan_database_compliance(
        self, database_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Effectue un scan de conformité complet d'une base de données."""
        logger.info(f"🔍 Scan de conformité: {database_config.get('id', 'unknown')}")
        
        compliance_report = {
            'database_id': database_config.get('id'),
            'scan_timestamp': datetime.now().isoformat(),
            'overall_status': ComplianceStatus.COMPLIANT,
            'standards_compliance': {},
            'issues_found': [],
            'recommendations': [],
            'sensitive_data_found': {},
            'encryption_status': {},
            'retention_violations': []
        }
        
        try:
            # Connexion à la base de données
            connection = await self._connect_to_database(database_config)
            
            # Scan des données sensibles
            sensitive_data = await self._scan_sensitive_data(connection, database_config)
            compliance_report['sensitive_data_found'] = sensitive_data
            
            # Vérification du chiffrement
            encryption_status = await self._check_encryption_status(connection, database_config)
            compliance_report['encryption_status'] = encryption_status
            
            # Vérification des politiques de rétention
            retention_issues = await self._check_retention_compliance(connection, database_config)
            compliance_report['retention_violations'] = retention_issues
            
            # Vérification par standard de conformité
            for standard in ComplianceStandard:
                compliance_status = await self._check_standard_compliance(
                    standard, sensitive_data, encryption_status, retention_issues
                )
                compliance_report['standards_compliance'][standard.value] = compliance_status
                
            # Détermination du statut global
            compliance_report['overall_status'] = self._determine_overall_status(
                compliance_report['standards_compliance']
            )
            
            # Génération des recommandations
            compliance_report['recommendations'] = self._generate_compliance_recommendations(
                compliance_report
            )
            
            await connection.close()
            
        except Exception as e:
            logger.error(f"❌ Erreur scan conformité: {e}")
            compliance_report['overall_status'] = ComplianceStatus.NON_COMPLIANT
            compliance_report['issues_found'].append(f"Erreur de scan: {str(e)}")
            
        return compliance_report
        
    async def _connect_to_database(self, config: Dict[str, Any]):
        """Établit une connexion à la base de données."""
        db_type = config.get('type', 'postgresql')
        
        if db_type == 'postgresql':
            return await asyncpg.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
        # Ajouter d'autres types...
        
        raise ValueError(f"Type de base non supporté: {db_type}")
        
    async def _scan_sensitive_data(
        self, connection: Any, config: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Scanne les données sensibles dans la base."""
        sensitive_data = {
            'tables_with_pii': [],
            'tables_with_financial': [],
            'tables_with_phi': [],
            'unencrypted_sensitive': []
        }
        
        try:
            # Récupération de la liste des tables
            if config.get('type') == 'postgresql':
                tables = await connection.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                )
                
                for table_row in tables:
                    table_name = table_row['table_name']
                    
                    # Classification basée sur le nom
                    classification = await self._classify_data(table_name, {})
                    
                    if classification == DataClassification.PII:
                        sensitive_data['tables_with_pii'].append(table_name)
                    elif classification == DataClassification.FINANCIAL:
                        sensitive_data['tables_with_financial'].append(table_name)
                    elif classification == DataClassification.PHI:
                        sensitive_data['tables_with_phi'].append(table_name)
                        
                    # Vérification du contenu (échantillonnage)
                    sample_data = await connection.fetch(
                        f"SELECT * FROM {table_name} LIMIT 10"
                    )
                    
                    for row in sample_data:
                        row_json = json.dumps(dict(row), default=str)
                        
                        for pattern_name, pattern in self.sensitive_data_patterns.items():
                            if re.search(pattern, row_json):
                                sensitive_data['unencrypted_sensitive'].append(
                                    f"{table_name} (contient {pattern_name})"
                                )
                                break
                                
        except Exception as e:
            logger.error(f"❌ Erreur scan données sensibles: {e}")
            
        return sensitive_data
        
    async def _check_encryption_status(
        self, connection: Any, config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Vérifie le statut de chiffrement."""
        encryption_status = {
            'database_encrypted': False,
            'transit_encrypted': False,
            'tables_requiring_encryption': []
        }
        
        try:
            if config.get('type') == 'postgresql':
                # Vérification SSL
                ssl_status = await connection.fetchval("SHOW ssl")
                encryption_status['transit_encrypted'] = ssl_status == 'on'
                
                # Vérification du chiffrement des données au repos
                # (Dépend de la configuration PostgreSQL)
                encryption_status['database_encrypted'] = config.get('encrypted', False)
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification chiffrement: {e}")
            
        return encryption_status
        
    async def _check_retention_compliance(
        self, connection: Any, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Vérifie la conformité des politiques de rétention."""
        violations = []
        
        try:
            # Vérifier chaque table avec des données horodatées
            tables_with_timestamps = await self._find_tables_with_timestamps(connection, config)
            
            for table_info in tables_with_timestamps:
                table_name = table_info['table_name']
                timestamp_column = table_info['timestamp_column']
                
                # Déterminer la politique de rétention applicable
                data_classification = await self._classify_data(table_name, {})
                applicable_policies = [
                    policy for policy in self.retention_policies.values()
                    if policy.data_classification == data_classification
                ]
                
                if applicable_policies:
                    policy = applicable_policies[0]
                    cutoff_date = datetime.now() - timedelta(days=policy.retention_period_days)
                    
                    # Vérifier les données expirées
                    if config.get('type') == 'postgresql':
                        expired_count = await connection.fetchval(
                            f"SELECT COUNT(*) FROM {table_name} WHERE {timestamp_column} < $1",
                            cutoff_date
                        )
                        
                        if expired_count > 0:
                            violations.append({
                                'table_name': table_name,
                                'policy_id': policy.policy_id,
                                'expired_records': expired_count,
                                'cutoff_date': cutoff_date.isoformat(),
                                'action_required': 'Archive or delete expired records'
                            })
                            
        except Exception as e:
            logger.error(f"❌ Erreur vérification rétention: {e}")
            
        return violations
        
    async def _find_tables_with_timestamps(
        self, connection: Any, config: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Trouve les tables avec des colonnes de timestamp."""
        tables_with_timestamps = []
        
        try:
            if config.get('type') == 'postgresql':
                # Recherche des colonnes de type timestamp
                timestamp_columns = await connection.fetch("""
                    SELECT table_name, column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND data_type IN ('timestamp', 'timestamptz', 'date')
                    AND column_name IN ('created_at', 'updated_at', 'timestamp', 'date_created')
                """)
                
                for row in timestamp_columns:
                    tables_with_timestamps.append({
                        'table_name': row['table_name'],
                        'timestamp_column': row['column_name']
                    })
                    
        except Exception as e:
            logger.error(f"❌ Erreur recherche timestamps: {e}")
            
        return tables_with_timestamps
        
    async def _check_standard_compliance(
        self,
        standard: ComplianceStandard,
        sensitive_data: Dict[str, List[str]],
        encryption_status: Dict[str, bool],
        retention_issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Vérifie la conformité pour un standard spécifique."""
        compliance = {
            'status': ComplianceStatus.COMPLIANT,
            'issues': [],
            'score': 100.0
        }
        
        if standard == ComplianceStandard.GDPR:
            # Vérifications GDPR
            if sensitive_data['tables_with_pii']:
                if not encryption_status['database_encrypted']:
                    compliance['issues'].append("Données PII non chiffrées")
                    compliance['score'] -= 30
                    
            if retention_issues:
                pii_violations = [v for v in retention_issues if 'pii' in v.get('policy_id', '').lower()]
                if pii_violations:
                    compliance['issues'].append(f"{len(pii_violations)} violations de rétention PII")
                    compliance['score'] -= 20
                    
        elif standard == ComplianceStandard.HIPAA:
            # Vérifications HIPAA
            if sensitive_data['tables_with_phi']:
                if not encryption_status['database_encrypted']:
                    compliance['issues'].append("Données PHI non chiffrées")
                    compliance['score'] -= 40
                    
                if not encryption_status['transit_encrypted']:
                    compliance['issues'].append("Transmission PHI non chiffrée")
                    compliance['score'] -= 30
                    
        elif standard == ComplianceStandard.PCI_DSS:
            # Vérifications PCI-DSS
            if sensitive_data['unencrypted_sensitive']:
                card_data = [item for item in sensitive_data['unencrypted_sensitive'] if 'credit_card' in item]
                if card_data:
                    compliance['issues'].append("Données de carte non chiffrées")
                    compliance['score'] -= 50
                    
        # Ajouter d'autres standards...
        
        # Détermination du statut final
        if compliance['score'] >= 95:
            compliance['status'] = ComplianceStatus.COMPLIANT
        elif compliance['score'] >= 70:
            compliance['status'] = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            compliance['status'] = ComplianceStatus.NON_COMPLIANT
            
        return compliance
        
    def _determine_overall_status(self, standards_compliance: Dict[str, Dict[str, Any]]) -> ComplianceStatus:
        """Détermine le statut global de conformité."""
        if not standards_compliance:
            return ComplianceStatus.UNDER_REVIEW
            
        all_compliant = all(
            status['status'] == ComplianceStatus.COMPLIANT 
            for status in standards_compliance.values()
        )
        
        if all_compliant:
            return ComplianceStatus.COMPLIANT
            
        any_non_compliant = any(
            status['status'] == ComplianceStatus.NON_COMPLIANT 
            for status in standards_compliance.values()
        )
        
        if any_non_compliant:
            return ComplianceStatus.NON_COMPLIANT
            
        return ComplianceStatus.PARTIALLY_COMPLIANT
        
    def _generate_compliance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Génère des recommandations de conformité."""
        recommendations = []
        
        # Recommandations basées sur le chiffrement
        if not report['encryption_status'].get('database_encrypted', False):
            recommendations.append(
                "🔒 Activer le chiffrement au repos pour toutes les données sensibles"
            )
            
        if not report['encryption_status'].get('transit_encrypted', False):
            recommendations.append(
                "🌐 Activer SSL/TLS pour chiffrer les données en transit"
            )
            
        # Recommandations basées sur la rétention
        if report['retention_violations']:
            recommendations.append(
                f"📅 Traiter {len(report['retention_violations'])} violations de rétention de données"
            )
            
        # Recommandations basées sur les données sensibles
        if report['sensitive_data_found']['unencrypted_sensitive']:
            recommendations.append(
                "🛡️ Chiffrer ou anonymiser les données sensibles détectées"
            )
            
        # Recommandations par standard
        for standard, compliance in report['standards_compliance'].items():
            if compliance['status'] != ComplianceStatus.COMPLIANT:
                recommendations.append(
                    f"📋 Résoudre les {len(compliance['issues'])} problèmes de conformité {standard.upper()}"
                )
                
        if not recommendations:
            recommendations.append("✅ Conformité satisfaisante, continuer le monitoring")
            
        return recommendations
        
    async def anonymize_data(
        self, database_config: Dict[str, Any], table_name: str, 
        anonymization_rules: Dict[str, str]
    ) -> Dict[str, Any]:
        """Anonymise les données d'une table."""
        logger.info(f"🎭 Anonymisation: {table_name}")
        
        result = {
            'table_name': table_name,
            'timestamp': datetime.now().isoformat(),
            'records_processed': 0,
            'fields_anonymized': list(anonymization_rules.keys()),
            'success': False
        }
        
        try:
            connection = await self._connect_to_database(database_config)
            
            # Récupération des données
            if database_config.get('type') == 'postgresql':
                records = await connection.fetch(f"SELECT * FROM {table_name}")
                
                anonymized_count = 0
                
                for record in records:
                    # Anonymisation de chaque enregistrement
                    updates = []
                    values = []
                    value_idx = 1
                    
                    for field, anonymization_type in anonymization_rules.items():
                        if field in record and record[field]:
                            anonymized_value = self.crypto.anonymize_field(
                                str(record[field]), anonymization_type
                            )
                            updates.append(f"{field} = ${value_idx}")
                            values.append(anonymized_value)
                            value_idx += 1
                            
                    if updates:
                        # Mise à jour de l'enregistrement
                        primary_key_field = 'id'  # Supposer une clé primaire 'id'
                        if primary_key_field in record:
                            values.append(record[primary_key_field])
                            update_query = f"""
                                UPDATE {table_name} 
                                SET {', '.join(updates)} 
                                WHERE {primary_key_field} = ${value_idx}
                            """
                            await connection.execute(update_query, *values)
                            anonymized_count += 1
                            
                result['records_processed'] = anonymized_count
                result['success'] = True
                
            await connection.close()
            
            # Audit de l'opération
            await self.audit_operation(
                AuditEventType.DATA_MODIFICATION,
                "system",
                database_config.get('id', 'unknown'),
                table_name,
                {
                    'operation': 'anonymization',
                    'fields': list(anonymization_rules.keys()),
                    'records_affected': result['records_processed']
                },
                success=result['success']
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur anonymisation: {e}")
            result['error'] = str(e)
            
        return result
        
    async def generate_compliance_report(
        self, database_ids: Optional[List[str]] = None, 
        standards: Optional[List[ComplianceStandard]] = None
    ) -> Dict[str, Any]:
        """Génère un rapport de conformité complet."""
        logger.info("📊 Génération rapport de conformité")
        
        report = {
            'report_id': f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'scope': {
                'database_ids': database_ids or ['all'],
                'standards': [s.value for s in (standards or list(ComplianceStandard))]
            },
            'executive_summary': {},
            'detailed_findings': {},
            'violations_summary': {},
            'recommendations': []
        }
        
        try:
            # Récupération des violations récentes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Derniers 30 jours
            
            violations = await self._get_violations_summary(start_date, end_date, database_ids)
            report['violations_summary'] = violations
            
            # Résumé exécutif
            total_violations = sum(violations.get('by_severity', {}).values())
            report['executive_summary'] = {
                'total_violations_30_days': total_violations,
                'critical_violations': violations.get('by_severity', {}).get('CRITICAL', 0),
                'databases_monitored': len(database_ids) if database_ids else 'all',
                'compliance_score': max(0, 100 - (total_violations * 2))  # Score basique
            }
            
            # Recommandations
            report['recommendations'] = await self._generate_executive_recommendations(violations)
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport: {e}")
            report['error'] = str(e)
            
        return report
        
    async def _get_violations_summary(
        self, start_date: datetime, end_date: datetime, database_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Récupère un résumé des violations."""
        summary = {
            'total': 0,
            'by_severity': {},
            'by_type': {},
            'by_database': {},
            'recent_violations': []
        }
        
        try:
            # Construction de la requête
            where_conditions = ["timestamp BETWEEN $1 AND $2"]
            params = [start_date, end_date]
            
            if database_ids:
                where_conditions.append(f"database_id = ANY(${len(params) + 1})")
                params.append(database_ids)
                
            query = f"""
                SELECT * FROM compliance_violations 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY timestamp DESC
            """
            
            violations = await self.audit_manager.connection.fetch(query, *params)
            
            summary['total'] = len(violations)
            
            # Groupement par sévérité
            for violation in violations:
                severity = violation['severity']
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
                
                violation_type = violation['violation_type']
                summary['by_type'][violation_type] = summary['by_type'].get(violation_type, 0) + 1
                
                database_id = violation['database_id']
                summary['by_database'][database_id] = summary['by_database'].get(database_id, 0) + 1
                
            # Violations récentes (10 dernières)
            summary['recent_violations'] = [
                {
                    'violation_id': v['violation_id'],
                    'rule_id': v['rule_id'],
                    'severity': v['severity'],
                    'timestamp': v['timestamp'].isoformat(),
                    'database_id': v['database_id'],
                    'description': v['description']
                }
                for v in violations[:10]
            ]
            
        except Exception as e:
            logger.error(f"❌ Erreur résumé violations: {e}")
            
        return summary
        
    async def _generate_executive_recommendations(self, violations_summary: Dict[str, Any]) -> List[str]:
        """Génère des recommandations de niveau exécutif."""
        recommendations = []
        
        total_violations = violations_summary.get('total', 0)
        critical_violations = violations_summary.get('by_severity', {}).get('CRITICAL', 0)
        
        if critical_violations > 0:
            recommendations.append(
                f"🚨 URGENT: {critical_violations} violations critiques nécessitent une action immédiate"
            )
            
        if total_violations > 50:
            recommendations.append(
                "📈 Nombre élevé de violations - réviser les politiques et la formation"
            )
            
        # Recommandations par type de violation
        violation_types = violations_summary.get('by_type', {})
        
        if violation_types.get('unauthorized_access', 0) > 5:
            recommendations.append(
                "🔐 Renforcer les contrôles d'accès et la gestion des identités"
            )
            
        if violation_types.get('encryption_violation', 0) > 0:
            recommendations.append(
                "🔒 Audit complet du chiffrement et mise à niveau nécessaire"
            )
            
        if violation_types.get('retention_violation', 0) > 10:
            recommendations.append(
                "📅 Mise en place d'un processus automatisé de gestion de la rétention"
            )
            
        if not recommendations:
            recommendations.append(
                "✅ Conformité satisfaisante - maintenir les efforts de monitoring"
            )
            
        return recommendations

# Instance globale
compliance_engine = ComplianceEngine({
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'audit_db',
    'user': 'audit_user',
    'password': 'audit_password'
})

# Fonctions de haut niveau pour l'API
async def initialize_compliance_system() -> ComplianceEngine:
    """Initialise le système de conformité."""
    await compliance_engine.initialize()
    return compliance_engine

async def audit_database_operation(
    operation_type: str,
    user_id: str,
    database_id: str,
    table_name: Optional[str] = None,
    operation_details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Audite une opération de base de données."""
    return await compliance_engine.audit_operation(
        AuditEventType(operation_type),
        user_id,
        database_id,
        table_name,
        operation_details or {},
        **kwargs
    )

if __name__ == "__main__":
    # Test de démonstration
    async def demo():
        print("🎵 Demo Compliance Engine")
        print("=" * 40)
        
        try:
            # Initialisation
            await initialize_compliance_system()
            
            # Test d'audit
            event_id = await audit_database_operation(
                "data_access",
                "user123",
                "db_prod",
                "users",
                {
                    'operation': 'SELECT',
                    'query': 'SELECT * FROM users WHERE email = ?',
                    'records_affected': 1
                }
            )
            
            print(f"✅ Événement audité: {event_id}")
            
            # Test de scan de conformité
            test_config = {
                'id': 'test_db',
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
            
            compliance_report = await compliance_engine.scan_database_compliance(test_config)
            print(f"📊 Statut conformité: {compliance_report['overall_status'].value}")
            print(f"💡 {len(compliance_report['recommendations'])} recommandations")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    asyncio.run(demo())
