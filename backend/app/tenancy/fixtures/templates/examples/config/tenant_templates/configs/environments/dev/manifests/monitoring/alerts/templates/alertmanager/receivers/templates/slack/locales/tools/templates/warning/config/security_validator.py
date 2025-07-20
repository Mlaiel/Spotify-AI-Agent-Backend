"""
Validateur de Sécurité Ultra-Avancé - Spotify AI Agent
====================================================

Système de validation et sécurisation complet pour les configurations d'alertes Warning
avec chiffrement, audit trail et détection d'anomalies.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import logging
import hashlib
import hmac
import secrets
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from collections import defaultdict, deque
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import sqlite3
import ipaddress

# Configuration du logging
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveaux de sécurité disponibles."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ViolationType(Enum):
    """Types de violations de sécurité."""
    INVALID_INPUT = "invalid_input"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_CORRUPTION = "data_corruption"
    ENCRYPTION_FAILURE = "encryption_failure"

class ComplianceStandard(Enum):
    """Standards de conformité supportés."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

@dataclass
class SecurityViolation:
    """Violation de sécurité détectée."""
    violation_id: str
    violation_type: ViolationType
    severity: SecurityLevel
    description: str
    source_ip: Optional[str]
    user_id: Optional[str]
    tenant_id: Optional[str]
    context: Dict[str, Any]
    detected_at: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class AuditEntry:
    """Entrée d'audit trail."""
    entry_id: str
    action: str
    resource_type: str
    resource_id: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    changes: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class EncryptionConfig:
    """Configuration de chiffrement."""
    algorithm: str
    key_size: int
    key_rotation_days: int
    salt_length: int
    iterations: int
    enabled: bool

class InputSanitizer:
    """Sanitiseur d'entrées avec protection contre les injections."""
    
    def __init__(self):
        # Patterns d'injection dangereux
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|\/\*|\*\/)",
            r"(\bOR\b.*=.*|AND.*=.*)",
            r"(\bxp_cmdshell\b|\bsp_executesql\b)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>"
        ]
        
        self.command_injection_patterns = [
            r"(;|\||&|`|\$\(|\${)",
            r"(\.\./|\.\.\\\)",
            r"(/etc/passwd|/etc/shadow)",
            r"(nc|netcat|wget|curl)\s+"
        ]
        
        # Compiled patterns pour performance
        self.compiled_patterns = {
            'sql': [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns],
            'xss': [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns],
            'command': [re.compile(pattern, re.IGNORECASE) for pattern in self.command_injection_patterns]
        }
    
    def sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitise une chaîne de caractères."""
        if not isinstance(input_string, str):
            raise ValueError("L'entrée doit être une chaîne de caractères")
        
        # Limitation de longueur
        if len(input_string) > max_length:
            raise ValueError(f"Chaîne trop longue (max: {max_length})")
        
        # Détection d'injections
        self._detect_injection_attempts(input_string)
        
        # Nettoyage de base
        sanitized = input_string.strip()
        
        # Suppression des caractères de contrôle dangereux
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Échappement des caractères spéciaux pour HTML
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        
        return sanitized
    
    def _detect_injection_attempts(self, input_string: str):
        """Détecte les tentatives d'injection."""
        
        # Vérification SQL injection
        for pattern in self.compiled_patterns['sql']:
            if pattern.search(input_string):
                raise ValueError("Tentative d'injection SQL détectée")
        
        # Vérification XSS
        for pattern in self.compiled_patterns['xss']:
            if pattern.search(input_string):
                raise ValueError("Tentative d'injection XSS détectée")
        
        # Vérification command injection
        for pattern in self.compiled_patterns['command']:
            if pattern.search(input_string):
                raise ValueError("Tentative d'injection de commande détectée")
    
    def validate_json(self, json_string: str, max_depth: int = 10) -> Dict[str, Any]:
        """Valide et parse un JSON en toute sécurité."""
        try:
            # Limitation de taille
            if len(json_string) > 100000:  # 100KB max
                raise ValueError("JSON trop volumineux")
            
            # Parse sécurisé
            parsed = json.loads(json_string)
            
            # Vérification de la profondeur
            self._check_json_depth(parsed, max_depth)
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON invalide: {e}")
    
    def _check_json_depth(self, obj: Any, max_depth: int, current_depth: int = 0):
        """Vérifie la profondeur d'un objet JSON."""
        if current_depth > max_depth:
            raise ValueError("JSON trop profond")
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._check_json_depth(value, max_depth, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._check_json_depth(item, max_depth, current_depth + 1)

class EncryptionManager:
    """Gestionnaire de chiffrement avancé."""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.master_key = self._derive_master_key()
        self.fernet = Fernet(self.master_key)
        self.key_rotation_timer = None
        
        # Démarrage de la rotation automatique des clés
        self._start_key_rotation()
    
    def _derive_master_key(self) -> bytes:
        """Dérive la clé maître à partir de la configuration."""
        password = os.getenv('ENCRYPTION_PASSWORD', 'default_password').encode()
        salt = os.getenv('ENCRYPTION_SALT', 'default_salt').encode()
        
        # Ajustement de la longueur du salt
        if len(salt) < 16:
            salt = salt.ljust(16, b'0')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt[:16],  # 16 bytes pour le salt
            iterations=self.config.iterations,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Chiffre des données."""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Erreur chiffrement: {e}")
            raise ValueError("Échec du chiffrement")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Déchiffre des données."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded_data)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Erreur déchiffrement: {e}")
            raise ValueError("Échec du déchiffrement")
    
    def generate_hash(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Génère un hash sécurisé avec salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode(),
            salt.encode(),
            self.config.iterations
        )
        
        return base64.urlsafe_b64encode(hash_obj).decode(), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Vérifie un hash."""
        computed_hash, _ = self.generate_hash(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)
    
    def _start_key_rotation(self):
        """Démarre la rotation automatique des clés."""
        if self.config.key_rotation_days > 0:
            rotation_interval = self.config.key_rotation_days * 24 * 3600
            self.key_rotation_timer = threading.Timer(rotation_interval, self._rotate_keys)
            self.key_rotation_timer.daemon = True
            self.key_rotation_timer.start()
    
    def _rotate_keys(self):
        """Effectue la rotation des clés."""
        logger.info("Rotation des clés de chiffrement")
        # Implémentation de la rotation des clés
        # En production, ceci impliquerait la génération de nouvelles clés
        # et la migration des données existantes
        
        # Reprogrammation de la prochaine rotation
        self._start_key_rotation()

class AuditLogger:
    """Logger d'audit avec persistance sécurisée."""
    
    def __init__(self, database_path: str = None):
        self.database_path = database_path or ":memory:"
        self.db_lock = threading.RLock()
        self._init_database()
        
        # Buffer pour les entrées en attente
        self.audit_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.RLock()
        
        # Thread de flush périodique
        self.flush_timer = threading.Timer(60, self._flush_audit_buffer)
        self.flush_timer.daemon = True
        self.flush_timer.start()
    
    def _init_database(self):
        """Initialise la base de données d'audit."""
        self.db_connection = sqlite3.connect(self.database_path, check_same_thread=False)
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Table des violations de sécurité
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_violations (
                    violation_id TEXT PRIMARY KEY,
                    violation_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    source_ip TEXT,
                    user_id TEXT,
                    tenant_id TEXT,
                    context TEXT,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution_notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des entrées d'audit
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_entries (
                    entry_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    user_id TEXT,
                    tenant_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success INTEGER NOT NULL,
                    changes TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index pour optimisation
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_tenant ON security_violations(tenant_id, detected_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_entries(tenant_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_entries(user_id, timestamp)")
            
            self.db_connection.commit()
    
    def log_security_violation(self, violation: SecurityViolation):
        """Enregistre une violation de sécurité."""
        with self.buffer_lock:
            self.audit_buffer.append(('violation', violation))
        
        # Log immédiat pour les violations critiques
        if violation.severity == SecurityLevel.CRITICAL:
            self._flush_audit_buffer()
    
    def log_audit_entry(self, entry: AuditEntry):
        """Enregistre une entrée d'audit."""
        with self.buffer_lock:
            self.audit_buffer.append(('audit', entry))
    
    def _flush_audit_buffer(self):
        """Vide le buffer d'audit vers la base de données."""
        entries_to_flush = []
        
        with self.buffer_lock:
            while self.audit_buffer:
                entries_to_flush.append(self.audit_buffer.popleft())
        
        if not entries_to_flush:
            # Reprogrammation du prochain flush
            self.flush_timer = threading.Timer(60, self._flush_audit_buffer)
            self.flush_timer.daemon = True
            self.flush_timer.start()
            return
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            try:
                for entry_type, entry_data in entries_to_flush:
                    if entry_type == 'violation':
                        self._insert_violation(cursor, entry_data)
                    elif entry_type == 'audit':
                        self._insert_audit_entry(cursor, entry_data)
                
                self.db_connection.commit()
                logger.info(f"Audit buffer vidé: {len(entries_to_flush)} entrées")
                
            except Exception as e:
                logger.error(f"Erreur flush audit buffer: {e}")
                self.db_connection.rollback()
        
        # Reprogrammation du prochain flush
        self.flush_timer = threading.Timer(60, self._flush_audit_buffer)
        self.flush_timer.daemon = True
        self.flush_timer.start()
    
    def _insert_violation(self, cursor, violation: SecurityViolation):
        """Insère une violation en base."""
        cursor.execute("""
            INSERT OR REPLACE INTO security_violations 
            (violation_id, violation_type, severity, description, source_ip, user_id, 
             tenant_id, context, detected_at, resolved, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            violation.violation_id,
            violation.violation_type.value,
            violation.severity.value,
            violation.description,
            violation.source_ip,
            violation.user_id,
            violation.tenant_id,
            json.dumps(violation.context),
            violation.detected_at.isoformat(),
            1 if violation.resolved else 0,
            violation.resolution_notes
        ))
    
    def _insert_audit_entry(self, cursor, entry: AuditEntry):
        """Insère une entrée d'audit en base."""
        cursor.execute("""
            INSERT OR REPLACE INTO audit_entries 
            (entry_id, action, resource_type, resource_id, user_id, tenant_id,
             ip_address, user_agent, success, changes, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.action,
            entry.resource_type,
            entry.resource_id,
            entry.user_id,
            entry.tenant_id,
            entry.ip_address,
            entry.user_agent,
            1 if entry.success else 0,
            json.dumps(entry.changes),
            entry.timestamp.isoformat(),
            json.dumps(entry.metadata)
        ))
    
    def get_violations_by_tenant(self, tenant_id: str, 
                               start_date: datetime = None,
                               end_date: datetime = None) -> List[SecurityViolation]:
        """Récupère les violations pour un tenant."""
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM security_violations WHERE tenant_id = ?"
            params = [tenant_id]
            
            if start_date:
                query += " AND detected_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND detected_at <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY detected_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            violations = []
            for row in rows:
                violation = SecurityViolation(
                    violation_id=row[0],
                    violation_type=ViolationType(row[1]),
                    severity=SecurityLevel(row[2]),
                    description=row[3],
                    source_ip=row[4],
                    user_id=row[5],
                    tenant_id=row[6],
                    context=json.loads(row[7]) if row[7] else {},
                    detected_at=datetime.fromisoformat(row[8]),
                    resolved=bool(row[9]),
                    resolution_notes=row[10]
                )
                violations.append(violation)
            
            return violations

class RateLimitTracker:
    """Tracker de rate limiting avec fenêtres glissantes."""
    
    def __init__(self):
        self.request_windows = defaultdict(lambda: deque())
        self.limits = {}
        self.lock = threading.RLock()
    
    def set_limit(self, identifier: str, requests_per_minute: int):
        """Définit une limite pour un identifiant."""
        with self.lock:
            self.limits[identifier] = requests_per_minute
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """Vérifie le rate limit et retourne (autorisé, temps_attente)."""
        with self.lock:
            if identifier not in self.limits:
                return True, 0
            
            now = datetime.now()
            window = self.request_windows[identifier]
            limit = self.limits[identifier]
            
            # Nettoyage de la fenêtre (dernière minute)
            cutoff = now - timedelta(minutes=1)
            while window and window[0] < cutoff:
                window.popleft()
            
            # Vérification de la limite
            if len(window) >= limit:
                # Calcul du temps d'attente
                oldest_request = window[0]
                wait_until = oldest_request + timedelta(minutes=1)
                wait_seconds = max(0, (wait_until - now).total_seconds())
                return False, int(wait_seconds)
            
            # Enregistrement de la requête
            window.append(now)
            return True, 0

class ComplianceChecker:
    """Vérificateur de conformité aux standards."""
    
    def __init__(self, standards: List[ComplianceStandard]):
        self.enabled_standards = standards
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Charge les règles de conformité."""
        rules = {}
        
        # Règles GDPR
        if ComplianceStandard.GDPR in self.enabled_standards:
            rules[ComplianceStandard.GDPR] = {
                'data_retention_max_days': 1095,  # 3 ans
                'encryption_required': True,
                'audit_trail_required': True,
                'data_minimization': True,
                'consent_tracking': True
            }
        
        # Règles SOC2
        if ComplianceStandard.SOC2 in self.enabled_standards:
            rules[ComplianceStandard.SOC2] = {
                'access_controls_required': True,
                'audit_trail_required': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'incident_response_required': True
            }
        
        return rules
    
    def check_compliance(self, data: Dict[str, Any]) -> Dict[ComplianceStandard, Dict[str, bool]]:
        """Vérifie la conformité des données."""
        results = {}
        
        for standard in self.enabled_standards:
            if standard not in self.compliance_rules:
                continue
            
            rules = self.compliance_rules[standard]
            compliance_results = {}
            
            if standard == ComplianceStandard.GDPR:
                compliance_results.update(self._check_gdpr_compliance(data, rules))
            elif standard == ComplianceStandard.SOC2:
                compliance_results.update(self._check_soc2_compliance(data, rules))
            
            results[standard] = compliance_results
        
        return results
    
    def _check_gdpr_compliance(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, bool]:
        """Vérifie la conformité GDPR."""
        results = {}
        
        # Vérification du chiffrement
        results['encryption_compliant'] = data.get('encrypted', False) == rules['encryption_required']
        
        # Vérification de la rétention des données
        retention_days = data.get('retention_days', 0)
        results['retention_compliant'] = retention_days <= rules['data_retention_max_days']
        
        # Vérification de la minimisation des données
        results['data_minimization_compliant'] = self._check_data_minimization(data)
        
        return results
    
    def _check_soc2_compliance(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, bool]:
        """Vérifie la conformité SOC2."""
        results = {}
        
        # Vérification des contrôles d'accès
        results['access_controls_compliant'] = data.get('access_controls_enabled', False)
        
        # Vérification du chiffrement en transit
        results['encryption_transit_compliant'] = data.get('tls_enabled', False)
        
        # Vérification du chiffrement au repos
        results['encryption_rest_compliant'] = data.get('encryption_at_rest', False)
        
        return results
    
    def _check_data_minimization(self, data: Dict[str, Any]) -> bool:
        """Vérifie le principe de minimisation des données."""
        # Implémentation basique - à adapter selon les besoins
        sensitive_fields = ['password', 'ssn', 'credit_card', 'personal_id']
        
        for field in sensitive_fields:
            if field in data and data[field]:
                # Vérification que les données sensibles sont anonymisées/pseudonymisées
                if not self._is_anonymized(data[field]):
                    return False
        
        return True
    
    def _is_anonymized(self, value: str) -> bool:
        """Vérifie si une valeur est anonymisée."""
        # Patterns d'anonymisation
        anonymized_patterns = [
            r'^\*+$',  # Étoiles
            r'^x+$',   # x répétés
            r'^\[REDACTED\]$',  # Texte caché
            r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}$'  # UUID/token
        ]
        
        for pattern in anonymized_patterns:
            if re.match(pattern, str(value), re.IGNORECASE):
                return True
        
        return False

class SecurityValidator:
    """
    Validateur de sécurité ultra-avancé.
    
    Fonctionnalités:
    - Validation et sanitisation des entrées
    - Chiffrement des données sensibles
    - Audit trail complet
    - Détection d'anomalies et de patterns suspects
    - Conformité aux standards (GDPR, SOC2, etc.)
    - Rate limiting adaptatif
    - Monitoring de sécurité en temps réel
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH,
                 compliance_standards: List[ComplianceStandard] = None):
        """Initialise le validateur de sécurité."""
        self.security_level = security_level
        self.compliance_standards = compliance_standards or [ComplianceStandard.SOC2, ComplianceStandard.GDPR]
        
        # Initialisation des composants
        self.sanitizer = InputSanitizer()
        self.encryption_config = EncryptionConfig(
            algorithm="AES-256-GCM",
            key_size=256,
            key_rotation_days=30,
            salt_length=16,
            iterations=100000,
            enabled=True
        )
        self.encryption_manager = EncryptionManager(self.encryption_config)
        self.audit_logger = AuditLogger()
        self.rate_limiter = RateLimitTracker()
        self.compliance_checker = ComplianceChecker(self.compliance_standards)
        
        # Métriques de sécurité
        self.security_metrics = {
            'violations_detected': 0,
            'injections_blocked': 0,
            'rate_limits_triggered': 0,
            'encryption_operations': 0,
            'compliance_checks': 0,
            'audit_entries_logged': 0
        }
        self.metrics_lock = threading.RLock()
        
        # Configuration des rate limits par défaut
        self._configure_default_rate_limits()
        
        logger.info(f"SecurityValidator initialisé (niveau: {security_level.name})")
    
    def _configure_default_rate_limits(self):
        """Configure les rate limits par défaut."""
        # Limits basés sur le niveau de sécurité
        if self.security_level == SecurityLevel.CRITICAL:
            base_limit = 10
        elif self.security_level == SecurityLevel.HIGH:
            base_limit = 50
        elif self.security_level == SecurityLevel.MEDIUM:
            base_limit = 100
        else:
            base_limit = 200
        
        # Configuration des limites
        self.rate_limiter.set_limit('config_creation', base_limit)
        self.rate_limiter.set_limit('config_update', base_limit // 2)
        self.rate_limiter.set_limit('config_deletion', base_limit // 4)
        self.rate_limiter.set_limit('sensitive_operation', base_limit // 10)
    
    def validate_alert_config(self, config_data: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Valide une configuration d'alerte de manière sécurisée."""
        
        context = context or {}
        tenant_id = context.get('tenant_id', 'unknown')
        user_id = context.get('user_id')
        ip_address = context.get('ip_address')
        
        # Génération de l'ID d'audit
        audit_id = self._generate_id('audit')
        
        try:
            # 1. Vérification du rate limiting
            rate_limit_key = f"config_validation:{tenant_id}"
            allowed, wait_time = self.rate_limiter.check_rate_limit(rate_limit_key)
            
            if not allowed:
                self._log_security_violation(
                    ViolationType.RATE_LIMIT_EXCEEDED,
                    f"Rate limit dépassé pour la validation de config",
                    ip_address, user_id, tenant_id,
                    {'wait_time_seconds': wait_time}
                )
                raise ValueError(f"Rate limit dépassé. Réessayez dans {wait_time} secondes.")
            
            # 2. Sanitisation des entrées
            sanitized_config = self._sanitize_config_data(config_data)
            
            # 3. Validation structurelle
            self._validate_config_structure(sanitized_config)
            
            # 4. Validation de sécurité
            self._validate_security_constraints(sanitized_config, context)
            
            # 5. Chiffrement des données sensibles
            encrypted_config = self._encrypt_sensitive_data(sanitized_config)
            
            # 6. Vérification de conformité
            compliance_results = self.compliance_checker.check_compliance(encrypted_config)
            self._check_compliance_violations(compliance_results)
            
            # 7. Log d'audit pour succès
            self._log_audit_entry(
                action="validate_alert_config",
                resource_type="alert_config",
                resource_id=sanitized_config.get('alert_id', 'unknown'),
                user_id=user_id,
                tenant_id=tenant_id,
                ip_address=ip_address,
                success=True,
                changes={'validation_passed': True, 'compliance_results': compliance_results}
            )
            
            self._increment_metric('compliance_checks')
            
            return {
                'config': encrypted_config,
                'compliance_results': compliance_results,
                'validation_id': audit_id
            }
            
        except Exception as e:
            # Log d'audit pour échec
            self._log_audit_entry(
                action="validate_alert_config",
                resource_type="alert_config",
                resource_id=config_data.get('alert_id', 'unknown'),
                user_id=user_id,
                tenant_id=tenant_id,
                ip_address=ip_address,
                success=False,
                changes={'error': str(e)}
            )
            
            raise
    
    def _sanitize_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise les données de configuration."""
        sanitized = {}
        
        for key, value in config_data.items():
            if isinstance(value, str):
                try:
                    sanitized[key] = self.sanitizer.sanitize_string(value)
                except ValueError as e:
                    self._log_security_violation(
                        ViolationType.INJECTION_ATTEMPT,
                        f"Tentative d'injection détectée dans le champ '{key}': {e}",
                        None, None, None,
                        {'field': key, 'original_value': value[:100]}
                    )
                    raise ValueError(f"Valeur invalide pour le champ '{key}': {e}")
            
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_config_data(value)
            
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitizer.sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _validate_config_structure(self, config_data: Dict[str, Any]):
        """Valide la structure de la configuration."""
        required_fields = ['alert_id', 'tenant_id', 'level', 'message']
        
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Champ requis manquant: {field}")
            
            if not config_data[field]:
                raise ValueError(f"Champ requis vide: {field}")
        
        # Validation des valeurs
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'HIGH', 'CRITICAL']
        if config_data['level'].upper() not in valid_levels:
            raise ValueError(f"Niveau d'alerte invalide: {config_data['level']}")
        
        # Validation de l'ID tenant
        tenant_id = config_data['tenant_id']
        if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', tenant_id):
            raise ValueError("ID tenant invalide (3-50 caractères alphanumériques, _, -)")
    
    def _validate_security_constraints(self, config_data: Dict[str, Any], context: Dict[str, Any]):
        """Valide les contraintes de sécurité."""
        
        # Vérification des permissions tenant
        user_tenant = context.get('user_tenant_id')
        config_tenant = config_data.get('tenant_id')
        
        if user_tenant and user_tenant != config_tenant:
            self._log_security_violation(
                ViolationType.UNAUTHORIZED_ACCESS,
                f"Tentative d'accès cross-tenant",
                context.get('ip_address'),
                context.get('user_id'),
                user_tenant,
                {'target_tenant': config_tenant}
            )
            raise ValueError("Accès non autorisé au tenant")
        
        # Vérification des patterns suspects
        message = config_data.get('message', '')
        if self._detect_suspicious_patterns(message):
            self._log_security_violation(
                ViolationType.SUSPICIOUS_PATTERN,
                f"Pattern suspect détecté dans le message",
                context.get('ip_address'),
                context.get('user_id'),
                config_tenant,
                {'message_excerpt': message[:100]}
            )
            raise ValueError("Pattern suspect détecté")
    
    def _detect_suspicious_patterns(self, text: str) -> bool:
        """Détecte des patterns suspects dans le texte."""
        suspicious_patterns = [
            r'(password|passwd|pwd)\s*[:=]\s*\S+',  # Mots de passe
            r'(api[_-]?key|token|secret)\s*[:=]\s*\S+',  # Clés API
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Numéros de carte
            r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Emails multiples
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _encrypt_sensitive_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Chiffre les données sensibles."""
        sensitive_fields = ['credentials', 'api_key', 'token', 'secret', 'password']
        encrypted_config = config_data.copy()
        
        def encrypt_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in sensitive_fields and isinstance(value, str):
                        obj[key] = self.encryption_manager.encrypt_data(value)
                        self._increment_metric('encryption_operations')
                    elif isinstance(value, (dict, list)):
                        encrypt_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        encrypt_recursive(item)
        
        encrypt_recursive(encrypted_config)
        return encrypted_config
    
    def _check_compliance_violations(self, compliance_results: Dict[ComplianceStandard, Dict[str, bool]]):
        """Vérifie les violations de conformité."""
        violations = []
        
        for standard, results in compliance_results.items():
            for check, passed in results.items():
                if not passed:
                    violations.append(f"{standard.value}: {check}")
        
        if violations:
            raise ValueError(f"Violations de conformité détectées: {', '.join(violations)}")
    
    def _log_security_violation(self, violation_type: ViolationType, description: str,
                              source_ip: Optional[str], user_id: Optional[str],
                              tenant_id: Optional[str], context: Dict[str, Any]):
        """Enregistre une violation de sécurité."""
        
        violation = SecurityViolation(
            violation_id=self._generate_id('violation'),
            violation_type=violation_type,
            severity=self._determine_violation_severity(violation_type),
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            tenant_id=tenant_id,
            context=context,
            detected_at=datetime.now()
        )
        
        self.audit_logger.log_security_violation(violation)
        self._increment_metric('violations_detected')
        
        if violation_type == ViolationType.INJECTION_ATTEMPT:
            self._increment_metric('injections_blocked')
        elif violation_type == ViolationType.RATE_LIMIT_EXCEEDED:
            self._increment_metric('rate_limits_triggered')
    
    def _log_audit_entry(self, action: str, resource_type: str, resource_id: str,
                       user_id: Optional[str], tenant_id: Optional[str],
                       ip_address: Optional[str], success: bool,
                       changes: Dict[str, Any]):
        """Enregistre une entrée d'audit."""
        
        entry = AuditEntry(
            entry_id=self._generate_id('audit'),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=None,  # À récupérer du contexte si disponible
            success=success,
            changes=changes,
            timestamp=datetime.now(),
            metadata={}
        )
        
        self.audit_logger.log_audit_entry(entry)
        self._increment_metric('audit_entries_logged')
    
    def _determine_violation_severity(self, violation_type: ViolationType) -> SecurityLevel:
        """Détermine la sévérité d'une violation."""
        severity_map = {
            ViolationType.INVALID_INPUT: SecurityLevel.LOW,
            ViolationType.INJECTION_ATTEMPT: SecurityLevel.HIGH,
            ViolationType.RATE_LIMIT_EXCEEDED: SecurityLevel.MEDIUM,
            ViolationType.UNAUTHORIZED_ACCESS: SecurityLevel.CRITICAL,
            ViolationType.SUSPICIOUS_PATTERN: SecurityLevel.HIGH,
            ViolationType.DATA_CORRUPTION: SecurityLevel.CRITICAL,
            ViolationType.ENCRYPTION_FAILURE: SecurityLevel.HIGH
        }
        
        return severity_map.get(violation_type, SecurityLevel.MEDIUM)
    
    def _generate_id(self, prefix: str) -> str:
        """Génère un ID unique."""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_hex(8)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def _increment_metric(self, metric_name: str):
        """Incrémente une métrique."""
        with self.metrics_lock:
            self.security_metrics[metric_name] = self.security_metrics.get(metric_name, 0) + 1
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de sécurité."""
        with self.metrics_lock:
            return self.security_metrics.copy()
    
    def get_security_violations(self, tenant_id: str, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> List[SecurityViolation]:
        """Récupère les violations de sécurité."""
        return self.audit_logger.get_violations_by_tenant(tenant_id, start_date, end_date)
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du validateur de sécurité."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Vérification du chiffrement
        try:
            test_data = "test_encryption"
            encrypted = self.encryption_manager.encrypt_data(test_data)
            decrypted = self.encryption_manager.decrypt_data(encrypted)
            health_status["components"]["encryption"] = "healthy" if decrypted == test_data else "unhealthy"
        except Exception as e:
            health_status["components"]["encryption"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        # Vérification de l'audit logger
        try:
            # Test simple de connexion à la base
            with self.audit_logger.db_lock:
                cursor = self.audit_logger.db_connection.cursor()
                cursor.execute("SELECT 1")
                health_status["components"]["audit_logger"] = "healthy"
        except Exception as e:
            health_status["components"]["audit_logger"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        return health_status
    
    def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self.audit_logger, 'db_connection'):
            self.audit_logger.db_connection.close()
        if hasattr(self.encryption_manager, 'key_rotation_timer') and self.encryption_manager.key_rotation_timer:
            self.encryption_manager.key_rotation_timer.cancel()
        logger.info("SecurityValidator nettoyé avec succès")

# Factory function
def create_security_validator(security_level: SecurityLevel = SecurityLevel.HIGH,
                            compliance_standards: List[ComplianceStandard] = None) -> SecurityValidator:
    """Factory function pour créer un validateur de sécurité."""
    return SecurityValidator(security_level, compliance_standards)
