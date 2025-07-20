"""
Système de sécurité et outils de conformité avancés.

Ce module fournit un système complet de sécurité avec audit, conformité,
chiffrement, détection d'intrusion et gestion des vulnérabilités.
"""

import os
import re
import json
import yaml
import hashlib
import secrets
import subprocess
import ssl
import socket
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import requests
import sqlite3


class SecurityLevel(str, Enum):
    """Niveaux de sécurité."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(str, Enum):
    """Types de vulnérabilités."""
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    ENCRYPTION = "encryption"
    CONFIGURATION = "configuration"
    ACCESS_CONTROL = "access_control"
    NETWORK = "network"
    DEPENDENCY = "dependency"


class ComplianceStandard(str, Enum):
    """Standards de conformité."""
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"


class AuditEventType(str, Enum):
    """Types d'événements d'audit."""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    MODIFICATION = "modification"
    DELETION = "deletion"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_CHECK = "compliance_check"


class SecurityPolicy(BaseModel):
    """Politique de sécurité."""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    compliance_standards: List[ComplianceStandard]
    severity: SecurityLevel
    enabled: bool = True
    auto_remediation: bool = False
    notification_channels: List[str] = Field(default_factory=list)


class VulnerabilityReport(BaseModel):
    """Rapport de vulnérabilité."""
    id: str
    title: str
    description: str
    vulnerability_type: VulnerabilityType
    severity: SecurityLevel
    affected_components: List[str]
    cve_ids: List[str] = Field(default_factory=list)
    remediation_steps: List[str]
    discovered_at: datetime
    fixed_at: Optional[datetime] = None
    status: str = "open"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditEvent(BaseModel):
    """Événement d'audit."""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    resource: str
    action: str
    result: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class EncryptionService:
    """Service de chiffrement avancé."""
    
    def __init__(self):
        """Initialise le service de chiffrement."""
        self.backend = default_backend()
    
    def generate_symmetric_key(self) -> bytes:
        """Génère une clé symétrique AES-256."""
        return secrets.token_bytes(32)  # 256 bits
    
    def generate_asymmetric_keypair(self, key_size: int = 2048) -> tuple:
        """Génère une paire de clés RSA."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def encrypt_symmetric(self, data: bytes, key: bytes) -> bytes:
        """Chiffre des données avec AES-256-GCM."""
        # Génération d'un IV aléatoire
        iv = secrets.token_bytes(12)  # 96 bits pour GCM
        
        # Chiffrement
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Retourne IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext
    
    def decrypt_symmetric(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Déchiffre des données avec AES-256-GCM."""
        # Extraction des composants
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Déchiffrement
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_asymmetric(self, data: bytes, public_key) -> bytes:
        """Chiffre des données avec RSA-OAEP."""
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes, private_key) -> bytes:
        """Déchiffre des données avec RSA-OAEP."""
        return private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple:
        """Hash un mot de passe avec Argon2."""
        try:
            import argon2
            
            if salt is None:
                salt = secrets.token_bytes(32)
            
            ph = argon2.PasswordHasher(
                time_cost=3,
                memory_cost=65536,
                parallelism=1,
                hash_len=32,
                salt_len=32
            )
            
            hash_value = ph.hash(password)
            return hash_value, salt
            
        except ImportError:
            # Fallback to PBKDF2 if argon2 not available
            import hashlib
            
            if salt is None:
                salt = secrets.token_bytes(32)
            
            hash_value = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000  # iterations
            )
            
            return hash_value.hex(), salt
    
    def verify_password(self, password: str, hash_value: str) -> bool:
        """Vérifie un mot de passe."""
        try:
            import argon2
            
            ph = argon2.PasswordHasher()
            try:
                ph.verify(hash_value, password)
                return True
            except argon2.exceptions.VerifyMismatchError:
                return False
                
        except ImportError:
            # Fallback logic for PBKDF2
            # Note: This would need the original salt stored separately
            return False
    
    def secure_random_string(self, length: int = 32) -> str:
        """Génère une chaîne aléatoire sécurisée."""
        return secrets.token_urlsafe(length)
    
    def generate_salt(self, length: int = 32) -> bytes:
        """Génère un salt cryptographiquement sûr."""
        return secrets.token_bytes(length)


class VulnerabilityScanner:
    """Scanner de vulnérabilités."""
    
    def __init__(self):
        """Initialise le scanner."""
        self.encryption_service = EncryptionService()
        self.vulnerability_db: Dict[str, VulnerabilityReport] = {}
    
    async def scan_network_ports(self, target: str, ports: List[int] = None) -> Dict[str, Any]:
        """Scanne les ports réseau d'une cible."""
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 443, 993, 995]
        
        results = {
            'target': target,
            'scan_time': datetime.now(),
            'open_ports': [],
            'closed_ports': [],
            'security_issues': []
        }
        
        for port in ports:
            try:
                # Test de connexion avec timeout
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((target, port))
                sock.close()
                
                if result == 0:
                    results['open_ports'].append(port)
                    
                    # Vérification de services à risque
                    if port in [21, 23, 80]:  # FTP, Telnet, HTTP non chiffré
                        results['security_issues'].append(f"Port {port} non sécurisé ouvert")
                else:
                    results['closed_ports'].append(port)
                    
            except Exception as e:
                print(f"Erreur scan port {port}: {e}")
        
        return results
    
    async def scan_ssl_certificates(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """Scanne les certificats SSL/TLS."""
        results = {
            'hostname': hostname,
            'port': port,
            'scan_time': datetime.now(),
            'certificate_valid': False,
            'security_issues': [],
            'certificate_info': {}
        }
        
        try:
            # Création du contexte SSL
            context = ssl.create_default_context()
            
            # Connexion et récupération du certificat
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    if cert:
                        results['certificate_valid'] = True
                        results['certificate_info'] = {
                            'subject': dict(x[0] for x in cert['subject']),
                            'issuer': dict(x[0] for x in cert['issuer']),
                            'version': cert['version'],
                            'serial_number': cert['serialNumber'],
                            'not_before': cert['notBefore'],
                            'not_after': cert['notAfter']
                        }
                        
                        # Vérification de l'expiration
                        expiry_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (expiry_date - datetime.now()).days
                        
                        if days_until_expiry < 30:
                            results['security_issues'].append(
                                f"Certificat expire dans {days_until_expiry} jours"
                            )
                        
                        # Vérification de l'algorithme de signature
                        if 'sha1' in cert.get('signatureAlgorithm', '').lower():
                            results['security_issues'].append("Algorithme de signature SHA-1 obsolète")
        
        except Exception as e:
            results['security_issues'].append(f"Erreur SSL: {e}")
        
        return results
    
    async def scan_dependencies(self, requirements_file: str) -> Dict[str, Any]:
        """Scanne les dépendances pour les vulnérabilités connues."""
        results = {
            'file': requirements_file,
            'scan_time': datetime.now(),
            'total_dependencies': 0,
            'vulnerable_dependencies': [],
            'security_issues': []
        }
        
        try:
            # Lecture du fichier de requirements
            with open(requirements_file, 'r') as f:
                dependencies = f.readlines()
            
            results['total_dependencies'] = len(dependencies)
            
            # Base de données simplifiée de vulnérabilités
            # En production, utiliser une vraie base de données de vulnérabilités
            vulnerable_packages = {
                'django': ['<3.2.13', '<4.0.4'],
                'flask': ['<2.2.0'],
                'requests': ['<2.25.1'],
                'pillow': ['<8.3.2'],
                'urllib3': ['<1.26.5']
            }
            
            for dep in dependencies:
                dep = dep.strip()
                if dep and not dep.startswith('#'):
                    # Parsing de la dépendance
                    package_name = dep.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    
                    if package_name.lower() in vulnerable_packages:
                        results['vulnerable_dependencies'].append({
                            'package': package_name,
                            'current': dep,
                            'vulnerable_versions': vulnerable_packages[package_name.lower()]
                        })
                        results['security_issues'].append(
                            f"Dépendance potentiellement vulnérable: {package_name}"
                        )
        
        except Exception as e:
            results['security_issues'].append(f"Erreur scan dépendances: {e}")
        
        return results
    
    async def scan_file_permissions(self, directory: str) -> Dict[str, Any]:
        """Scanne les permissions de fichiers."""
        results = {
            'directory': directory,
            'scan_time': datetime.now(),
            'total_files': 0,
            'security_issues': [],
            'suspicious_files': []
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    results['total_files'] += 1
                    
                    try:
                        stat = os.stat(file_path)
                        mode = stat.st_mode
                        
                        # Vérification des permissions dangereuses
                        if mode & 0o002:  # World writable
                            results['suspicious_files'].append(f"{file_path}: écriture publique")
                            results['security_issues'].append(
                                f"Fichier avec écriture publique: {file_path}"
                            )
                        
                        if mode & 0o004 and file.endswith(('.key', '.pem', '.p12')):  # Readable key files
                            results['suspicious_files'].append(f"{file_path}: clé lisible")
                            results['security_issues'].append(
                                f"Fichier de clé lisible: {file_path}"
                            )
                        
                    except Exception as e:
                        print(f"Erreur permissions {file_path}: {e}")
        
        except Exception as e:
            results['security_issues'].append(f"Erreur scan permissions: {e}")
        
        return results
    
    def create_vulnerability_report(
        self,
        title: str,
        description: str,
        vuln_type: VulnerabilityType,
        severity: SecurityLevel,
        affected_components: List[str],
        remediation_steps: List[str]
    ) -> VulnerabilityReport:
        """Crée un rapport de vulnérabilité."""
        vuln_id = f"VULN-{datetime.now().strftime('%Y%m%d')}-{len(self.vulnerability_db) + 1:04d}"
        
        report = VulnerabilityReport(
            id=vuln_id,
            title=title,
            description=description,
            vulnerability_type=vuln_type,
            severity=severity,
            affected_components=affected_components,
            remediation_steps=remediation_steps,
            discovered_at=datetime.now()
        )
        
        self.vulnerability_db[vuln_id] = report
        return report


class ComplianceChecker:
    """Vérificateur de conformité."""
    
    def __init__(self):
        """Initialise le vérificateur."""
        self.policies: Dict[str, SecurityPolicy] = {}
        self.compliance_results: Dict[str, Dict[str, Any]] = {}
    
    def add_policy(self, policy: SecurityPolicy):
        """Ajoute une politique de sécurité."""
        self.policies[policy.name] = policy
    
    async def check_gdpr_compliance(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Vérifie la conformité GDPR."""
        results = {
            'standard': 'GDPR',
            'check_time': datetime.now(),
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Vérification de la minimisation des données
        if 'data_retention' not in data_config:
            results['violations'].append("Politique de rétention des données manquante")
            results['compliant'] = False
        
        # Vérification du chiffrement
        if not data_config.get('encryption_enabled', False):
            results['violations'].append("Chiffrement des données personnelles non activé")
            results['compliant'] = False
        
        # Vérification des consentements
        if 'consent_management' not in data_config:
            results['violations'].append("Système de gestion des consentements manquant")
            results['compliant'] = False
        
        # Vérification du droit à l'oubli
        if not data_config.get('data_deletion_enabled', False):
            results['violations'].append("Mécanisme de suppression des données manquant")
            results['compliant'] = False
        
        # Vérification de la portabilité
        if 'data_export' not in data_config:
            results['recommendations'].append("Implémenter l'export des données utilisateur")
        
        return results
    
    async def check_soc2_compliance(self, security_config: Dict[str, Any]) -> Dict[str, Any]:
        """Vérifie la conformité SOC 2."""
        results = {
            'standard': 'SOC2',
            'check_time': datetime.now(),
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Vérification de la sécurité
        if not security_config.get('multi_factor_auth', False):
            results['violations'].append("Authentification multi-facteurs non activée")
            results['compliant'] = False
        
        # Vérification de la disponibilité
        if 'backup_strategy' not in security_config:
            results['violations'].append("Stratégie de sauvegarde manquante")
            results['compliant'] = False
        
        # Vérification de l'intégrité de traitement
        if not security_config.get('data_validation', False):
            results['violations'].append("Validation de l'intégrité des données manquante")
            results['compliant'] = False
        
        # Vérification de la confidentialité
        if not security_config.get('access_controls', False):
            results['violations'].append("Contrôles d'accès insuffisants")
            results['compliant'] = False
        
        # Vérification de la vie privée
        if 'privacy_controls' not in security_config:
            results['recommendations'].append("Implémenter des contrôles de confidentialité")
        
        return results
    
    async def check_iso27001_compliance(self, isms_config: Dict[str, Any]) -> Dict[str, Any]:
        """Vérifie la conformité ISO 27001."""
        results = {
            'standard': 'ISO27001',
            'check_time': datetime.now(),
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Vérification de la politique de sécurité
        if 'security_policy' not in isms_config:
            results['violations'].append("Politique de sécurité de l'information manquante")
            results['compliant'] = False
        
        # Vérification de l'analyse des risques
        if 'risk_assessment' not in isms_config:
            results['violations'].append("Analyse des risques manquante")
            results['compliant'] = False
        
        # Vérification de la formation
        if not isms_config.get('security_training', False):
            results['violations'].append("Programme de formation à la sécurité manquant")
            results['compliant'] = False
        
        # Vérification de la gestion des incidents
        if 'incident_response' not in isms_config:
            results['violations'].append("Plan de réponse aux incidents manquant")
            results['compliant'] = False
        
        # Vérification de l'audit interne
        if not isms_config.get('internal_audit', False):
            results['recommendations'].append("Mettre en place un audit interne régulier")
        
        return results
    
    async def run_compliance_check(self, standard: ComplianceStandard, config: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute une vérification de conformité."""
        if standard == ComplianceStandard.GDPR:
            return await self.check_gdpr_compliance(config)
        elif standard == ComplianceStandard.SOC2:
            return await self.check_soc2_compliance(config)
        elif standard == ComplianceStandard.ISO27001:
            return await self.check_iso27001_compliance(config)
        else:
            return {
                'standard': standard.value,
                'check_time': datetime.now(),
                'compliant': False,
                'violations': [f"Vérification pour {standard.value} non implémentée"],
                'recommendations': []
            }


class AuditLogger:
    """Logger d'audit sécurisé."""
    
    def __init__(self, audit_db_path: str):
        """Initialise le logger d'audit."""
        self.db_path = audit_db_path
        self.encryption_service = EncryptionService()
        self._init_database()
        
        # Génération d'une clé de chiffrement pour les logs sensibles
        self.log_encryption_key = self.encryption_service.generate_symmetric_key()
    
    def _init_database(self):
        """Initialise la base de données d'audit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    timestamp TIMESTAMP,
                    user_id TEXT,
                    resource TEXT,
                    action TEXT,
                    result TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    details_encrypted BLOB,
                    hash_chain TEXT
                )
            """)
            conn.commit()
    
    def log_event(
        self,
        event_type: AuditEventType,
        resource: str,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enregistre un événement d'audit."""
        event_id = f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
        
        event = AuditEvent(
            id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        # Chiffrement des détails sensibles
        details_json = json.dumps(event.details)
        encrypted_details = self.encryption_service.encrypt_symmetric(
            details_json.encode('utf-8'),
            self.log_encryption_key
        )
        
        # Calcul du hash pour l'intégrité
        hash_chain = self._calculate_hash_chain(event)
        
        # Sauvegarde en base
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_events 
                (id, event_type, timestamp, user_id, resource, action, result,
                 ip_address, user_agent, details_encrypted, hash_chain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.event_type.value,
                event.timestamp,
                event.user_id,
                event.resource,
                event.action,
                event.result,
                event.ip_address,
                event.user_agent,
                encrypted_details,
                hash_chain
            ))
            conn.commit()
        
        return event_id
    
    def _calculate_hash_chain(self, event: AuditEvent) -> str:
        """Calcule un hash chaîné pour l'intégrité."""
        # Récupération du dernier hash
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT hash_chain FROM audit_events ORDER BY timestamp DESC LIMIT 1"
            )
            last_hash = cursor.fetchone()
            last_hash = last_hash[0] if last_hash else "0"
        
        # Création du hash actuel
        event_data = f"{event.id}{event.timestamp}{event.resource}{event.action}{last_hash}"
        return hashlib.sha256(event_data.encode()).hexdigest()
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Récupère la piste d'audit."""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if resource:
            query += " AND resource = ?"
            params.append(resource)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                # Déchiffrement des détails
                try:
                    decrypted_details = self.encryption_service.decrypt_symmetric(
                        row[9],  # details_encrypted
                        self.log_encryption_key
                    )
                    details = json.loads(decrypted_details.decode('utf-8'))
                except Exception:
                    details = {}
                
                event = AuditEvent(
                    id=row[0],
                    event_type=AuditEventType(row[1]),
                    timestamp=datetime.fromisoformat(row[2]),
                    user_id=row[3],
                    resource=row[4],
                    action=row[5],
                    result=row[6],
                    ip_address=row[7],
                    user_agent=row[8],
                    details=details
                )
                events.append(event)
        
        return events
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Vérifie l'intégrité de la piste d'audit."""
        results = {
            'integrity_valid': True,
            'total_events': 0,
            'corrupted_events': [],
            'verification_time': datetime.now()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, resource, action, hash_chain FROM audit_events ORDER BY timestamp"
            )
            
            previous_hash = "0"
            for row in cursor.fetchall():
                event_id, timestamp, resource, action, stored_hash = row
                
                # Recalcul du hash
                event_data = f"{event_id}{timestamp}{resource}{action}{previous_hash}"
                calculated_hash = hashlib.sha256(event_data.encode()).hexdigest()
                
                if calculated_hash != stored_hash:
                    results['integrity_valid'] = False
                    results['corrupted_events'].append(event_id)
                
                previous_hash = stored_hash
                results['total_events'] += 1
        
        return results


class SecurityEngine:
    """Moteur de sécurité principal."""
    
    def __init__(self, storage_path: str):
        """Initialise le moteur de sécurité."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.encryption_service = EncryptionService()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.compliance_checker = ComplianceChecker()
        self.audit_logger = AuditLogger(str(self.storage_path / "audit.db"))
        
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.active_monitors: Dict[str, asyncio.Task] = {}
    
    def add_security_policy(self, policy: SecurityPolicy):
        """Ajoute une politique de sécurité."""
        self.security_policies[policy.name] = policy
        self.compliance_checker.add_policy(policy)
    
    async def run_security_scan(
        self,
        target: str,
        scan_types: List[str] = None
    ) -> Dict[str, Any]:
        """Exécute un scan de sécurité complet."""
        if scan_types is None:
            scan_types = ['network', 'ssl', 'dependencies', 'permissions']
        
        results = {
            'target': target,
            'scan_time': datetime.now(),
            'scan_types': scan_types,
            'results': {},
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'high_issues': 0,
                'medium_issues': 0,
                'low_issues': 0
            }
        }
        
        # Scan réseau
        if 'network' in scan_types:
            results['results']['network'] = await self.vulnerability_scanner.scan_network_ports(target)
        
        # Scan SSL
        if 'ssl' in scan_types:
            results['results']['ssl'] = await self.vulnerability_scanner.scan_ssl_certificates(target)
        
        # Scan dépendances
        if 'dependencies' in scan_types:
            requirements_file = f"{target}/requirements.txt"
            if os.path.exists(requirements_file):
                results['results']['dependencies'] = await self.vulnerability_scanner.scan_dependencies(requirements_file)
        
        # Scan permissions
        if 'permissions' in scan_types:
            if os.path.isdir(target):
                results['results']['permissions'] = await self.vulnerability_scanner.scan_file_permissions(target)
        
        # Calcul du résumé
        for scan_result in results['results'].values():
            issue_count = len(scan_result.get('security_issues', []))
            results['summary']['total_issues'] += issue_count
        
        # Log de l'événement d'audit
        self.audit_logger.log_event(
            AuditEventType.SECURITY_INCIDENT,
            target,
            "security_scan",
            "completed",
            details={'scan_types': scan_types, 'issues_found': results['summary']['total_issues']}
        )
        
        return results
    
    async def run_compliance_audit(
        self,
        standards: List[ComplianceStandard],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Exécute un audit de conformité."""
        results = {
            'audit_time': datetime.now(),
            'standards': [s.value for s in standards],
            'overall_compliant': True,
            'results': {}
        }
        
        for standard in standards:
            compliance_result = await self.compliance_checker.run_compliance_check(standard, config)
            results['results'][standard.value] = compliance_result
            
            if not compliance_result['compliant']:
                results['overall_compliant'] = False
        
        # Log de l'audit
        self.audit_logger.log_event(
            AuditEventType.COMPLIANCE_CHECK,
            "system",
            "compliance_audit",
            "completed" if results['overall_compliant'] else "violations_found",
            details={'standards': [s.value for s in standards]}
        )
        
        return results
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Génère un rapport de sécurité."""
        return {
            'report_time': datetime.now(),
            'policies_count': len(self.security_policies),
            'vulnerabilities_count': len(self.vulnerability_scanner.vulnerability_db),
            'audit_integrity': self.audit_logger.verify_integrity(),
            'recent_events': self.audit_logger.get_audit_trail(limit=50)
        }


# Politiques de sécurité par défaut
DEFAULT_SECURITY_POLICIES = [
    SecurityPolicy(
        name="password_policy",
        description="Politique de mots de passe robustes",
        rules=[
            {"type": "min_length", "value": 12},
            {"type": "require_uppercase", "value": True},
            {"type": "require_lowercase", "value": True},
            {"type": "require_numbers", "value": True},
            {"type": "require_special_chars", "value": True}
        ],
        compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001],
        severity=SecurityLevel.HIGH
    ),
    SecurityPolicy(
        name="encryption_policy",
        description="Politique de chiffrement des données",
        rules=[
            {"type": "encryption_at_rest", "value": True},
            {"type": "encryption_in_transit", "value": True},
            {"type": "key_rotation", "value": "quarterly"}
        ],
        compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOC2],
        severity=SecurityLevel.CRITICAL
    ),
    SecurityPolicy(
        name="access_control_policy",
        description="Politique de contrôle d'accès",
        rules=[
            {"type": "principle_least_privilege", "value": True},
            {"type": "regular_access_review", "value": "monthly"},
            {"type": "multi_factor_auth", "value": True}
        ],
        compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001],
        severity=SecurityLevel.HIGH
    )
]


# Factory functions
def create_security_engine(storage_path: str = "/var/security") -> SecurityEngine:
    """Crée un moteur de sécurité."""
    engine = SecurityEngine(storage_path)
    
    # Ajout des politiques par défaut
    for policy in DEFAULT_SECURITY_POLICIES:
        engine.add_security_policy(policy)
    
    return engine


def create_encryption_service() -> EncryptionService:
    """Crée un service de chiffrement."""
    return EncryptionService()


async def setup_security_monitoring(target: str) -> SecurityEngine:
    """Configure un monitoring de sécurité de base."""
    engine = create_security_engine()
    
    # Exécution d'un scan initial
    await engine.run_security_scan(target)
    
    return engine
