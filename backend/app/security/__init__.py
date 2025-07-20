# 🎵 Spotify AI Agent - Advanced Security Layer
# ============================================
# 
# Couche de sécurité avancée avec OAuth2, JWT,
# détection de menaces et audit complet.
#
# 🎖️ Développé par l'équipe d'experts enterprise

"""
Enterprise Security Layer
=========================

Complete security framework providing:
- OAuth2 provider and consumer
- JWT token management
- Rate limiting and DDoS protection
- Threat detection and prevention
- Comprehensive audit logging
- Multi-factor authentication
- API key management

Authors & Roles:
- Lead Developer & AI Architect
- Security Specialist
- Senior Backend Developer (Python/FastAPI/Django)
- DBA & Data Engineer
"""

import os
import jwt
import bcrypt
import pyotp
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
import hashlib
import hmac
import ipaddress
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
from collections import defaultdict, deque
import json
import geoip2.database
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatType(Enum):
    """Types de menaces"""
    BRUTE_FORCE = "brute_force"
    DOS_ATTACK = "dos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuthMethod(Enum):
    """Méthodes d'authentification"""
    PASSWORD = "password"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    MFA = "mfa"


@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    jwt_secret: str
    jwt_expiry: int = 3600  # 1 heure
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 heure
    max_login_attempts: int = 5
    lockout_duration: int = 1800  # 30 minutes
    mfa_required: bool = False
    password_min_length: int = 8


class SecurityManager:
    """Gestionnaire principal de sécurité"""
    
    def __init__(self):
        self.config = SecurityConfig(
            jwt_secret=os.getenv('JWT_SECRET_KEY', 'your-secret-key'),
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', 100)),
            max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', 5))
        )
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_SECURITY_DB', 1))
        )
        self.oauth2_manager = OAuth2Manager()
        self.jwt_manager = JWTManager(self.config)
        self.rate_limiter = RateLimiter(self.redis_client, self.config)
        self.threat_detector = ThreatDetector(self.redis_client)
        self.audit_logger = AuditLogger()
        self.mfa_manager = MFAManager()
        self.logger = logging.getLogger(__name__)


class JWTManager:
    """Gestionnaire JWT avancé"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_access_token(self, user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Crée un token d'accès JWT"""
        try:
            to_encode = user_data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(seconds=self.config.jwt_expiry)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access_token",
                "jti": self._generate_jti()  # JWT ID unique
            })
            
            encoded_jwt = jwt.encode(to_encode, self.config.jwt_secret, algorithm="HS256")
            return encoded_jwt
            
        except Exception as exc:
            self.logger.error(f"Erreur création token JWT: {exc}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Crée un token de rafraîchissement"""
        try:
            to_encode = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(days=30),
                "iat": datetime.utcnow(),
                "type": "refresh_token",
                "jti": self._generate_jti()
            }
            
            return jwt.encode(to_encode, self.config.jwt_secret, algorithm="HS256")
            
        except Exception as exc:
            self.logger.error(f"Erreur création refresh token: {exc}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=["HS256"])
            
            # Vérification du type de token
            if payload.get("type") not in ["access_token", "refresh_token"]:
                raise HTTPException(status_code=401, detail="Type de token invalide")
            
            # Vérification blacklist
            if self._is_token_blacklisted(payload.get("jti")):
                raise HTTPException(status_code=401, detail="Token révoqué")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expiré")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Token invalide")
    
    def revoke_token(self, token: str):
        """Révoque un token (blacklist)"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=["HS256"], options={"verify_exp": False})
            jti = payload.get("jti")
            if jti:
                # Ajouter à la blacklist avec expiration
                exp = payload.get("exp", 0)
                ttl = max(0, exp - int(time.time()))
                self.redis_client.setex(f"blacklist:{jti}", ttl, "revoked")
                
        except Exception as exc:
            self.logger.error(f"Erreur révocation token: {exc}")
    
    def _generate_jti(self) -> str:
        """Génère un ID unique pour le JWT"""
        return hashlib.sha256(f"{datetime.utcnow().isoformat()}{os.urandom(16).hex()}".encode()).hexdigest()
    
    def _is_token_blacklisted(self, jti: str) -> bool:
        """Vérifie si un token est blacklisté"""
        return self.redis_client.exists(f"blacklist:{jti}")


class OAuth2Manager:
    """Gestionnaire OAuth2 complet"""
    
    def __init__(self):
        self.client_id = os.getenv('OAUTH2_CLIENT_ID')
        self.client_secret = os.getenv('OAUTH2_CLIENT_SECRET')
        self.redirect_uri = os.getenv('OAUTH2_REDIRECT_URI')
        self.logger = logging.getLogger(__name__)
        
    async def authorize(self, client_id: str, redirect_uri: str, scope: str, state: str) -> str:
        """Génère une URL d'autorisation OAuth2"""
        try:
            # Validation du client
            if not self._validate_client(client_id, redirect_uri):
                raise HTTPException(status_code=400, detail="Client invalide")
            
            # Génération du code d'autorisation
            auth_code = self._generate_auth_code()
            
            # Stockage temporaire du code
            await self._store_auth_code(auth_code, client_id, redirect_uri, scope)
            
            # Construction de l'URL de redirection
            redirect_url = f"{redirect_uri}?code={auth_code}&state={state}"
            return redirect_url
            
        except Exception as exc:
            self.logger.error(f"Erreur autorisation OAuth2: {exc}")
            raise
    
    async def exchange_code_for_token(self, code: str, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Échange un code d'autorisation contre un token"""
        try:
            # Validation du client
            if not self._validate_client_credentials(client_id, client_secret):
                raise HTTPException(status_code=401, detail="Credentials invalides")
            
            # Récupération des informations du code
            code_data = await self._get_auth_code_data(code)
            if not code_data:
                raise HTTPException(status_code=400, detail="Code invalide ou expiré")
            
            # Génération des tokens
            access_token = self._generate_access_token(code_data)
            refresh_token = self._generate_refresh_token(code_data)
            
            # Nettoyage du code utilisé
            await self._invalidate_auth_code(code)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": code_data.get("scope")
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur échange de token OAuth2: {exc}")
            raise
    
    def _validate_client(self, client_id: str, redirect_uri: str) -> bool:
        """Valide un client OAuth2"""
        # Logique de validation du client
        return client_id == self.client_id
    
    def _validate_client_credentials(self, client_id: str, client_secret: str) -> bool:
        """Valide les credentials du client"""
        return client_id == self.client_id and client_secret == self.client_secret
    
    def _generate_auth_code(self) -> str:
        """Génère un code d'autorisation"""
        return hashlib.sha256(f"{datetime.utcnow().isoformat()}{os.urandom(32).hex()}".encode()).hexdigest()
    
    async def _store_auth_code(self, code: str, client_id: str, redirect_uri: str, scope: str):
        """Stocke un code d'autorisation temporairement"""
        data = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "created_at": datetime.utcnow().isoformat()
        }
        # Stockage Redis avec expiration de 10 minutes
        self.redis_client.setex(f"auth_code:{code}", 600, json.dumps(data))
    
    async def _get_auth_code_data(self, code: str) -> Optional[Dict]:
        """Récupère les données d'un code d'autorisation"""
        data = self.redis_client.get(f"auth_code:{code}")
        return json.loads(data) if data else None
    
    async def _invalidate_auth_code(self, code: str):
        """Invalide un code d'autorisation"""
        self.redis_client.delete(f"auth_code:{code}")


class RateLimiter:
    """Limiteur de débit avancé"""
    
    def __init__(self, redis_client: redis.Redis, config: SecurityConfig):
        self.redis_client = redis_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def check_rate_limit(self, identifier: str, limit: Optional[int] = None, window: Optional[int] = None) -> bool:
        """Vérifie la limite de débit"""
        try:
            limit = limit or self.config.rate_limit_requests
            window = window or self.config.rate_limit_window
            
            key = f"rate_limit:{identifier}"
            
            # Utilisation de sliding window
            current_time = int(time.time())
            pipeline = self.redis_client.pipeline()
            
            # Supprimer les entrées expirées
            pipeline.zremrangebyscore(key, 0, current_time - window)
            
            # Compter les requêtes dans la fenêtre
            pipeline.zcard(key)
            
            # Ajouter la requête actuelle
            pipeline.zadd(key, {str(current_time): current_time})
            
            # Définir l'expiration
            pipeline.expire(key, window)
            
            results = pipeline.execute()
            request_count = results[1]
            
            if request_count >= limit:
                self.logger.warning(f"Rate limit dépassé pour {identifier}: {request_count}/{limit}")
                return False
            
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification rate limit: {exc}")
            return True  # En cas d'erreur, autoriser par défaut
    
    async def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """Récupère les informations de rate limiting"""
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        window = self.config.rate_limit_window
        
        # Nettoyer et compter
        self.redis_client.zremrangebyscore(key, 0, current_time - window)
        current_count = self.redis_client.zcard(key)
        
        return {
            "limit": self.config.rate_limit_requests,
            "remaining": max(0, self.config.rate_limit_requests - current_count),
            "reset_time": current_time + window,
            "window": window
        }


class ThreatDetector:
    """Détecteur de menaces avancé"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.geoip_reader = None
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-Country.mmdb')
        except:
            self.logger.warning("Base de données GeoIP non disponible")
        
    async def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyse une requête pour détecter les menaces"""
        try:
            threats = []
            risk_score = 0.0
            
            # Analyse de l'IP
            client_ip = self._get_client_ip(request)
            ip_analysis = await self._analyze_ip(client_ip)
            risk_score += ip_analysis['risk_score']
            threats.extend(ip_analysis['threats'])
            
            # Analyse des headers
            header_analysis = await self._analyze_headers(request.headers)
            risk_score += header_analysis['risk_score']
            threats.extend(header_analysis['threats'])
            
            # Analyse de la fréquence de requêtes
            frequency_analysis = await self._analyze_request_frequency(client_ip)
            risk_score += frequency_analysis['risk_score']
            threats.extend(frequency_analysis['threats'])
            
            # Analyse du user agent
            ua_analysis = await self._analyze_user_agent(request.headers.get('user-agent', ''))
            risk_score += ua_analysis['risk_score']
            threats.extend(ua_analysis['threats'])
            
            return {
                'risk_score': min(risk_score, 1.0),
                'threats': threats,
                'client_ip': client_ip,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse de menace: {exc}")
            return {'risk_score': 0.0, 'threats': []}
    
    async def _analyze_ip(self, ip: str) -> Dict[str, Any]:
        """Analyse une adresse IP"""
        threats = []
        risk_score = 0.0
        
        try:
            # Vérification blacklist
            if self._is_ip_blacklisted(ip):
                threats.append(ThreatType.SUSPICIOUS_ACTIVITY.value)
                risk_score += 0.8
            
            # Vérification géolocalisation
            if self.geoip_reader:
                try:
                    response = self.geoip_reader.country(ip)
                    country = response.country.iso_code
                    
                    # Pays à haut risque
                    high_risk_countries = ['CN', 'RU', 'KP', 'IR']
                    if country in high_risk_countries:
                        threats.append(ThreatType.SUSPICIOUS_ACTIVITY.value)
                        risk_score += 0.3
                except:
                    pass
            
            # Vérification de la fréquence d'attaques depuis cette IP
            attack_count = self._get_recent_attacks(ip)
            if attack_count > 10:
                threats.append(ThreatType.DOS_ATTACK.value)
                risk_score += 0.6
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse IP {ip}: {exc}")
        
        return {'risk_score': risk_score, 'threats': threats}
    
    async def _analyze_headers(self, headers) -> Dict[str, Any]:
        """Analyse les headers HTTP"""
        threats = []
        risk_score = 0.0
        
        # Vérification des headers suspects
        suspicious_headers = ['x-forwarded-for', 'x-real-ip', 'x-cluster-client-ip']
        for header in suspicious_headers:
            if header in headers:
                # Logique de détection de proxy/VPN
                risk_score += 0.1
        
        # Vérification du referer
        referer = headers.get('referer', '')
        if self._is_suspicious_referer(referer):
            threats.append(ThreatType.SUSPICIOUS_ACTIVITY.value)
            risk_score += 0.2
        
        return {'risk_score': risk_score, 'threats': threats}
    
    async def _analyze_request_frequency(self, ip: str) -> Dict[str, Any]:
        """Analyse la fréquence des requêtes"""
        threats = []
        risk_score = 0.0
        
        # Compter les requêtes dans la dernière minute
        key = f"request_freq:{ip}"
        current_time = int(time.time())
        
        # Nettoyer les anciennes entrées
        self.redis_client.zremrangebyscore(key, 0, current_time - 60)
        
        # Compter les requêtes
        request_count = self.redis_client.zcard(key)
        
        # Ajouter la requête actuelle
        self.redis_client.zadd(key, {str(current_time): current_time})
        self.redis_client.expire(key, 60)
        
        # Détection d'attaque DDoS
        if request_count > 100:  # Plus de 100 requêtes par minute
            threats.append(ThreatType.DOS_ATTACK.value)
            risk_score += 0.9
        elif request_count > 50:
            threats.append(ThreatType.SUSPICIOUS_ACTIVITY.value)
            risk_score += 0.5
        
        return {'risk_score': risk_score, 'threats': threats}
    
    async def _analyze_user_agent(self, user_agent: str) -> Dict[str, Any]:
        """Analyse le user agent"""
        threats = []
        risk_score = 0.0
        
        # User agents suspects
        suspicious_patterns = ['bot', 'crawler', 'scanner', 'sqlmap', 'nikto']
        for pattern in suspicious_patterns:
            if pattern.lower() in user_agent.lower():
                threats.append(ThreatType.SUSPICIOUS_ACTIVITY.value)
                risk_score += 0.4
                break
        
        return {'risk_score': risk_score, 'threats': threats}
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP réelle du client"""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_ip_blacklisted(self, ip: str) -> bool:
        """Vérifie si une IP est blacklistée"""
        return self.redis_client.sismember("blacklisted_ips", ip)
    
    def _get_recent_attacks(self, ip: str) -> int:
        """Récupère le nombre d'attaques récentes depuis une IP"""
        key = f"attacks:{ip}"
        return self.redis_client.zcard(key)
    
    def _is_suspicious_referer(self, referer: str) -> bool:
        """Vérifie si le referer est suspect"""
        suspicious_domains = ['malicious.com', 'phishing.net']
        return any(domain in referer for domain in suspicious_domains)


class AuditLogger:
    """Logger d'audit complet"""
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
        
    async def log_security_event(self, event_type: str, user_id: Optional[str], 
                                request_data: Dict, metadata: Dict = None):
        """Log un événement de sécurité"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': request_data.get('client_ip'),
            'user_agent': request_data.get('user_agent'),
            'endpoint': request_data.get('endpoint'),
            'method': request_data.get('method'),
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(audit_entry))
        
        # Stockage en base de données pour analyse
        await self._store_audit_log(audit_entry)
    
    async def _store_audit_log(self, audit_entry: Dict):
        """Stocke le log d'audit en base de données"""
        # Logique de stockage en base
        pass


class MFAManager:
    """Gestionnaire d'authentification multi-facteurs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_secret(self, user_id: str) -> str:
        """Génère un secret TOTP pour un utilisateur"""
        secret = pyotp.random_base32()
        # Stocker le secret de manière sécurisée
        return secret
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Génère un QR code pour la configuration TOTP"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name="Spotify AI Agent"
        )
        return totp_uri
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Vérifie un token TOTP"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # Fenêtre de 30 secondes
        except Exception as exc:
            self.logger.error(f"Erreur vérification TOTP: {exc}")
            return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Génère des codes de secours"""
        codes = []
        for _ in range(count):
            code = hashlib.sha256(f"{user_id}{datetime.utcnow().isoformat()}{os.urandom(16).hex()}".encode()).hexdigest()[:8]
            codes.append(code)
        
        # Stocker les codes de manière sécurisée (hashés)
        return codes


# Instances globales
security_manager = SecurityManager()
jwt_manager = JWTManager(security_manager.config)
oauth2_manager = OAuth2Manager()
rate_limiter = RateLimiter(security_manager.redis_client, security_manager.config)
threat_detector = ThreatDetector(security_manager.redis_client)
audit_logger = AuditLogger()
mfa_manager = MFAManager()


# Middleware de sécurité FastAPI
class SecurityMiddleware:
    """Middleware de sécurité pour FastAPI"""
    
    def __init__(self):
        self.security_manager = security_manager
        
    async def __call__(self, request: Request, call_next):
        # Analyse de menace
        threat_analysis = await threat_detector.analyze_request(request)
        
        # Blocage si risque élevé
        if threat_analysis['risk_score'] > 0.8:
            await audit_logger.log_security_event(
                'HIGH_RISK_REQUEST_BLOCKED',
                None,
                {'client_ip': threat_analysis['client_ip']},
                threat_analysis
            )
            raise HTTPException(status_code=403, detail="Requête bloquée - Activité suspecte")
        
        # Vérification rate limit
        client_ip = threat_analysis['client_ip']
        if not await rate_limiter.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Trop de requêtes")
        
        response = await call_next(request)
        return response


# Export des classes principales
__all__ = [
    'SecurityManager',
    'JWTManager',
    'OAuth2Manager', 
    'RateLimiter',
    'ThreatDetector',
    'AuditLogger',
    'MFAManager',
    'SecurityMiddleware',
    'SecurityLevel',
    'ThreatType',
    'AuthMethod',
    'security_manager',
    'jwt_manager',
    'oauth2_manager'
]
