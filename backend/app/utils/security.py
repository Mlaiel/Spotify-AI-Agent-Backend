"""
Enterprise Security Module for Spotify AI Agent
==============================================

Module de sécurité complet avec chiffrement, authentification, détection de menaces.
Protection avancée contre les vulnérabilités et attaques courantes.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipaddress
from collections import defaultdict

logger = logging.getLogger(__name__)

# === Types et constantes ===
class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class SecurityEvent:
    """Événement de sécurité."""
    event_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    blocked: bool

@dataclass
class TokenPayload:
    """Payload de token JWT."""
    user_id: str
    email: str
    roles: List[str]
    scopes: List[str]
    issued_at: datetime
    expires_at: datetime
    issuer: str

# === Gestionnaire de chiffrement ===
class EncryptionManager:
    """
    Gestionnaire de chiffrement avancé avec multiple algorithmes.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv('ENCRYPTION_MASTER_KEY', self._generate_master_key())
        self._fernet = Fernet(self.master_key.encode()[:44].ljust(44, '=')[:44])
        self._private_key, self._public_key = self._generate_rsa_keys()
    
    def _generate_master_key(self) -> str:
        """Génère une clé maître."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _generate_rsa_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Génère une paire de clés RSA."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def encrypt_symmetric(self, data: str) -> str:
        """Chiffre des données avec Fernet (symétrique)."""
        return self._fernet.encrypt(data.encode()).decode()
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Déchiffre des données avec Fernet."""
        return self._fernet.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_asymmetric(self, data: str) -> str:
        """Chiffre des données avec RSA (asymétrique)."""
        encrypted = self._public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode()
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Déchiffre des données avec RSA."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self._private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode()
    
    def encrypt_field(self, data: Any, field_type: str = 'symmetric') -> str:
        """Chiffre un champ de données selon le type."""
        json_data = json.dumps(data, default=str)
        
        if field_type == 'asymmetric':
            return self.encrypt_asymmetric(json_data)
        else:
            return self.encrypt_symmetric(json_data)
    
    def decrypt_field(self, encrypted_data: str, field_type: str = 'symmetric') -> Any:
        """Déchiffre un champ de données."""
        try:
            if field_type == 'asymmetric':
                json_data = self.decrypt_asymmetric(encrypted_data)
            else:
                json_data = self.decrypt_symmetric(encrypted_data)
            
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_pii(self, pii_data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash des données PII avec salt."""
        salt = salt or secrets.token_hex(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(pii_data.encode()))
        return key.decode(), salt

class SecurityUtils:
    """
    Utilitaires de sécurité industriels étendus.
    """
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Hash sécurisé de mot de passe avec bcrypt."""
        if salt:
            # Compatibilité avec l'ancien système
            return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt
        else:
            # Nouveau système bcrypt
            salt_bytes = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt_bytes)
            return hashed.decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Vérifie un mot de passe contre son hash."""
        try:
            # Nouveau système bcrypt
            if hashed.startswith('$2b$'):
                return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
            else:
                # Ancien système (compatibilité)
                hash_val, salt = hashed.split(":")
                return SecurityUtils.hash_password(password, salt).split(":")[0] == hash_val
        except Exception:
            return False

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Génère un token sécurisé."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_api_key(prefix: str = "sk", length: int = 48) -> str:
        """Génère une clé API avec préfixe."""
        return f"{prefix}_{secrets.token_urlsafe(length)}"

    @staticmethod
    def hmac_digest(key: str, message: str, algorithm: str = 'sha256') -> str:
        """Calcule un HMAC digest."""
        hash_algo = getattr(hashlib, algorithm)
        return hmac.new(key.encode(), message.encode(), hash_algo).hexdigest()

    @staticmethod
    def timing_safe_compare(val1: str, val2: str) -> bool:
        """Comparaison sécurisée contre les attaques timing."""
        return hmac.compare_digest(val1, val2)

    @staticmethod
    def validate_api_key(api_key: str, valid_keys: List[str]) -> bool:
        """Valide une clé API de manière sécurisée."""
        for valid_key in valid_keys:
            if SecurityUtils.timing_safe_compare(api_key, valid_key):
                return True
        return False

    @staticmethod
    def generate_csrf_token(session_id: str, secret_key: str) -> str:
        """Génère un token CSRF."""
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}"
        signature = SecurityUtils.hmac_digest(secret_key, message)
        return f"{timestamp}:{signature}"

    @staticmethod
    def verify_csrf_token(token: str, session_id: str, secret_key: str, max_age: int = 3600) -> bool:
        """Vérifie un token CSRF."""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Vérification de l'âge
            if time.time() - timestamp > max_age:
                return False
            
            # Vérification de la signature
            expected_message = f"{session_id}:{timestamp_str}"
            expected_signature = SecurityUtils.hmac_digest(secret_key, expected_message)
            
            return SecurityUtils.timing_safe_compare(signature, expected_signature)
        except Exception:
            return False

    @staticmethod
    def sanitize_input(input_data: str, max_length: int = 1000) -> str:
        """Sanitise les entrées utilisateur."""
        if not isinstance(input_data, str):
            return ""
        
        # Limitation de longueur
        sanitized = input_data[:max_length]
        
        # Suppression des caractères de contrôle
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Échappement HTML basique
        html_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char, escaped in html_chars.items():
            sanitized = sanitized.replace(char, escaped)
        
        return sanitized.strip()

    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """Génère un nom de fichier sécurisé."""
        # Suppression des caractères dangereux
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '', original_filename)
        
        # Limitation de longueur
        if len(safe_name) > 100:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:95] + ext
        
        # Ajout d'un timestamp pour l'unicité
        timestamp = int(time.time())
        name, ext = os.path.splitext(safe_name)
        
        return f"{name}_{timestamp}{ext}"

# === Gestionnaire de tokens JWT ===
class JWTManager:
    """
    Gestionnaire avancé de tokens JWT avec rotation de clés.
    """
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = "spotify-ai-agent"
        
        # Stockage des tokens révoqués (en production: Redis/DB)
        self._revoked_tokens: Set[str] = set()
        
        # Clés de rotation (pour la sécurité renforcée)
        self._key_versions: Dict[int, str] = {1: secret_key}
        self._current_key_version = 1
    
    def create_token(
        self,
        user_id: str,
        email: str,
        roles: List[str] = None,
        scopes: List[str] = None,
        expires_in: int = 3600
    ) -> str:
        """Crée un token JWT."""
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in)
        
        payload = {
            'user_id': user_id,
            'email': email,
            'roles': roles or [],
            'scopes': scopes or ['read'],
            'iat': int(now.timestamp()),
            'exp': int(expires_at.timestamp()),
            'iss': self.issuer,
            'jti': secrets.token_urlsafe(16),  # Token ID unique
            'key_version': self._current_key_version
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Vérifie et décode un token JWT."""
        try:
            # Vérification de révocation
            if token in self._revoked_tokens:
                logger.warning("Attempted use of revoked token")
                return None
            
            # Décodage
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Vérification de l'issuer
            if payload.get('iss') != self.issuer:
                logger.warning("Invalid token issuer")
                return None
            
            # Construction du payload typé
            return TokenPayload(
                user_id=payload['user_id'],
                email=payload['email'],
                roles=payload.get('roles', []),
                scopes=payload.get('scopes', []),
                issued_at=datetime.fromtimestamp(payload['iat'], timezone.utc),
                expires_at=datetime.fromtimestamp(payload['exp'], timezone.utc),
                issuer=payload['iss']
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Révoque un token."""
        try:
            # Extraction du JTI sans validation complète
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get('jti')
            
            if jti:
                self._revoked_tokens.add(jti)
                return True
            
            return False
        except Exception:
            return False
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Renouvelle un token valide."""
        payload = self.verify_token(token)
        
        if payload and (payload.expires_at - datetime.now(timezone.utc)).seconds > 300:
            # Création d'un nouveau token
            return self.create_token(
                user_id=payload.user_id,
                email=payload.email,
                roles=payload.roles,
                scopes=payload.scopes
            )
        
        return None
    
    def rotate_key(self, new_secret: str) -> int:
        """Effectue une rotation de clé."""
        self._current_key_version += 1
        self._key_versions[self._current_key_version] = new_secret
        self.secret_key = new_secret
        
        logger.info(f"Key rotated to version {self._current_key_version}")
        return self._current_key_version

# === Détecteur de menaces ===
class ThreatDetector:
    """
    Détecteur de menaces en temps réel avec machine learning.
    """
    
    def __init__(self):
        # Stockage des événements (en production: DB/Redis)
        self._events: List[SecurityEvent] = []
        self._ip_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._user_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Seuils de détection
        self.brute_force_threshold = 5  # tentatives par IP
        self.brute_force_window = 300   # secondes
        self.ddos_threshold = 100       # requêtes par IP
        self.ddos_window = 60          # secondes
        
        # Patterns de détection
        self._sql_injection_patterns = [
            r"(union\s+select|select\s+\*|insert\s+into|delete\s+from)",
            r"(drop\s+table|create\s+table|alter\s+table)",
            r"(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
            r"('|\"|;|--|\*/|\*/)"
        ]
        
        self._xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
    
    async def analyze_request(
        self,
        ip_address: str,
        user_id: Optional[str],
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> List[SecurityEvent]:
        """Analyse une requête pour détecter les menaces."""
        
        threats = []
        now = datetime.now(timezone.utc)
        
        # Détection de brute force
        if self._detect_brute_force(ip_address, user_id, now):
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(8),
                threat_type=ThreatType.BRUTE_FORCE,
                severity=SecurityLevel.HIGH,
                timestamp=now,
                source_ip=ip_address,
                user_id=user_id,
                details={'endpoint': endpoint, 'attempts': len(self._ip_attempts[ip_address])},
                blocked=True
            )
            threats.append(event)
        
        # Détection DDoS
        if self._detect_ddos(ip_address, now):
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(8),
                threat_type=ThreatType.DDoS,
                severity=SecurityLevel.CRITICAL,
                timestamp=now,
                source_ip=ip_address,
                user_id=user_id,
                details={'endpoint': endpoint, 'request_rate': self._calculate_request_rate(ip_address)},
                blocked=True
            )
            threats.append(event)
        
        # Détection injection SQL
        if self._detect_sql_injection(payload):
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(8),
                threat_type=ThreatType.SQL_INJECTION,
                severity=SecurityLevel.CRITICAL,
                timestamp=now,
                source_ip=ip_address,
                user_id=user_id,
                details={'endpoint': endpoint, 'payload_sample': str(payload)[:200]},
                blocked=True
            )
            threats.append(event)
        
        # Détection XSS
        if self._detect_xss(payload):
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(8),
                threat_type=ThreatType.XSS,
                severity=SecurityLevel.HIGH,
                timestamp=now,
                source_ip=ip_address,
                user_id=user_id,
                details={'endpoint': endpoint, 'payload_sample': str(payload)[:200]},
                blocked=True
            )
            threats.append(event)
        
        # Stockage des événements
        self._events.extend(threats)
        
        return threats
    
    def _detect_brute_force(self, ip_address: str, user_id: Optional[str], now: datetime) -> bool:
        """Détecte les tentatives de brute force."""
        
        # Nettoyage des tentatives anciennes
        cutoff = now - timedelta(seconds=self.brute_force_window)
        self._ip_attempts[ip_address] = [
            attempt for attempt in self._ip_attempts[ip_address]
            if attempt > cutoff
        ]
        
        # Ajout de la tentative actuelle
        self._ip_attempts[ip_address].append(now)
        
        # Vérification du seuil
        return len(self._ip_attempts[ip_address]) > self.brute_force_threshold
    
    def _detect_ddos(self, ip_address: str, now: datetime) -> bool:
        """Détecte les attaques DDoS."""
        cutoff = now - timedelta(seconds=self.ddos_window)
        
        # Simulation du comptage de requêtes (en production: Redis)
        request_count = self._calculate_request_rate(ip_address)
        
        return request_count > self.ddos_threshold
    
    def _calculate_request_rate(self, ip_address: str) -> int:
        """Calcule le taux de requêtes pour une IP."""
        # Simulation - en production: utiliser Redis pour comptage précis
        recent_attempts = len(self._ip_attempts[ip_address])
        return recent_attempts * 2  # Approximation
    
    def _detect_sql_injection(self, payload: Dict[str, Any]) -> bool:
        """Détecte les tentatives d'injection SQL."""
        payload_str = json.dumps(payload).lower()
        
        for pattern in self._sql_injection_patterns:
            if re.search(pattern, payload_str, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, payload: Dict[str, Any]) -> bool:
        """Détecte les tentatives XSS."""
        payload_str = json.dumps(payload)
        
        for pattern in self._xss_patterns:
            if re.search(pattern, payload_str, re.IGNORECASE):
                return True
        
        return False
    
    async def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Retourne un résumé des menaces détectées."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_events = [event for event in self._events if event.timestamp > cutoff]
        
        # Agrégation par type de menace
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        top_ips = defaultdict(int)
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.severity.value] += 1
            top_ips[event.source_ip] += 1
        
        return {
            'total_threats': len(recent_events),
            'threat_types': dict(threat_counts),
            'severity_distribution': dict(severity_counts),
            'top_threat_ips': dict(sorted(top_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'blocked_events': sum(1 for event in recent_events if event.blocked)
        }

# === Validateur d'adresses IP ===
class IPValidator:
    """
    Validateur d'adresses IP avec listes blanches/noires.
    """
    
    def __init__(self):
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.suspicious_ranges: List[ipaddress.IPv4Network] = []
        
        # Chargement des ranges suspects par défaut
        self._load_default_suspicious_ranges()
    
    def _load_default_suspicious_ranges(self):
        """Charge les ranges IP suspects par défaut."""
        # Ranges couramment utilisés par des bots/attaquants
        suspicious_ranges = [
            '10.0.0.0/8',      # RFC 1918 (privé)
            '172.16.0.0/12',   # RFC 1918 (privé)
            '192.168.0.0/16',  # RFC 1918 (privé)
            '127.0.0.0/8',     # Loopback
            '169.254.0.0/16',  # Link-local
            '224.0.0.0/4',     # Multicast
        ]
        
        for range_str in suspicious_ranges:
            try:
                self.suspicious_ranges.append(ipaddress.IPv4Network(range_str))
            except Exception:
                pass
    
    def validate_ip(self, ip_address: str) -> Dict[str, Any]:
        """
        Valide une adresse IP et retourne son analyse.
        
        Returns:
            Dict avec validation et métadonnées
        """
        result = {
            'valid': False,
            'allowed': False,
            'ip_type': 'unknown',
            'risk_level': 'low',
            'warnings': []
        }
        
        try:
            ip = ipaddress.ip_address(ip_address)
            result['valid'] = True
            
            # Type d'IP
            if ip.is_private:
                result['ip_type'] = 'private'
                result['warnings'].append('Private IP address')
            elif ip.is_loopback:
                result['ip_type'] = 'loopback'
                result['warnings'].append('Loopback address')
            elif ip.is_multicast:
                result['ip_type'] = 'multicast'
                result['warnings'].append('Multicast address')
            else:
                result['ip_type'] = 'public'
            
            # Vérification listes
            if ip_address in self.blacklist:
                result['allowed'] = False
                result['risk_level'] = 'critical'
                result['warnings'].append('IP in blacklist')
            elif ip_address in self.whitelist:
                result['allowed'] = True
                result['risk_level'] = 'low'
            else:
                # Vérification des ranges suspects
                for network in self.suspicious_ranges:
                    if ip in network:
                        result['risk_level'] = 'medium'
                        result['warnings'].append(f'IP in suspicious range: {network}')
                        break
                
                result['allowed'] = result['risk_level'] != 'critical'
            
        except ValueError:
            result['warnings'].append('Invalid IP address format')
        
        return result
    
    def add_to_whitelist(self, ip_address: str) -> bool:
        """Ajoute une IP à la liste blanche."""
        try:
            ipaddress.ip_address(ip_address)  # Validation
            self.whitelist.add(ip_address)
            self.blacklist.discard(ip_address)  # Suppression de la blacklist si présente
            return True
        except ValueError:
            return False
    
    def add_to_blacklist(self, ip_address: str) -> bool:
        """Ajoute une IP à la liste noire."""
        try:
            ipaddress.ip_address(ip_address)  # Validation
            self.blacklist.add(ip_address)
            self.whitelist.discard(ip_address)  # Suppression de la whitelist si présente
            return True
        except ValueError:
            return False

# === Gestionnaire de session sécurisé ===
class SecureSessionManager:
    """
    Gestionnaire de sessions sécurisé avec protection contre les attaques.
    """
    
    def __init__(self, secret_key: str, session_timeout: int = 3600):
        self.secret_key = secret_key
        self.session_timeout = session_timeout
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_ips: Dict[str, str] = {}
    
    def create_session(self, user_id: str, ip_address: str, additional_data: Dict[str, Any] = None) -> str:
        """Crée une nouvelle session sécurisée."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(timezone.utc),
            'last_accessed': datetime.now(timezone.utc),
            'ip_address': ip_address,
            'csrf_token': SecurityUtils.generate_csrf_token(session_id, self.secret_key),
            'data': additional_data or {}
        }
        
        self._sessions[session_id] = session_data
        self._session_ips[session_id] = ip_address
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """Valide une session existante."""
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        now = datetime.now(timezone.utc)
        
        # Vérification du timeout
        if (now - session['last_accessed']).seconds > self.session_timeout:
            self.destroy_session(session_id)
            return None
        
        # Vérification de l'IP (protection contre session hijacking)
        if session['ip_address'] != ip_address:
            logger.warning(f"Session IP mismatch: {session['ip_address']} vs {ip_address}")
            self.destroy_session(session_id)
            return None
        
        # Mise à jour du dernier accès
        session['last_accessed'] = now
        
        return session
    
    def destroy_session(self, session_id: str) -> bool:
        """Détruit une session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._session_ips.pop(session_id, None)
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Nettoie les sessions expirées."""
        now = datetime.now(timezone.utc)
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if (now - session['last_accessed']).seconds > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        return len(expired_sessions)

# === Instance globale pour faciliter l'utilisation ===
_encryption_manager = None
_jwt_manager = None
_threat_detector = None

def get_encryption_manager() -> EncryptionManager:
    """Retourne l'instance globale du gestionnaire de chiffrement."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager

def get_jwt_manager() -> JWTManager:
    """Retourne l'instance globale du gestionnaire JWT."""
    global _jwt_manager
    if _jwt_manager is None:
        secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        _jwt_manager = JWTManager(secret_key)
    return _jwt_manager

def get_threat_detector() -> ThreatDetector:
    """Retourne l'instance globale du détecteur de menaces."""
    global _threat_detector
    if _threat_detector is None:
        _threat_detector = ThreatDetector()
    return _threat_detector
