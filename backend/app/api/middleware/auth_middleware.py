"""
Middleware d'authentification avancé pour Spotify AI Agent
Gestion complète : JWT, OAuth Spotify, sessions Redis, sécurité
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import redis.asyncio as redis
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import SecurityUtils
from app.utils.metrics_manager import MetricsManager
from app.core.logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer()
metrics = MetricsManager()


class AuthTokenData(BaseModel):
    """Données du token d'authentification"""
    user_id: str
    username: str
    email: str
    role: str = "user"
    spotify_id: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    session_id: str
    device_id: Optional[str] = None
    ip_address: str
    expires_at: datetime
    issued_at: datetime


class SpotifyAuthData(BaseModel):
    """Données d'authentification Spotify"""
    access_token: str
    refresh_token: str
    expires_at: datetime
    scope: List[str]
    user_id: str
    spotify_user_id: str


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware d'authentification principal
    - Gestion JWT
    - Sessions Redis
    - Limitation de taux
    - Logging sécurisé
    """
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        self.security_utils = SecurityUtils()
        
        # Endpoints publics (pas d'auth requise)
        self.public_endpoints = {
            "/docs", "/redoc", "/openapi.json",
            "/health", "/metrics", "/auth/login",
            "/auth/register", "/auth/spotify/callback"
        }
        
        # Endpoints admin uniquement
        self.admin_endpoints = {
            "/admin", "/users/all", "/system/config"
        }

    async def dispatch(self, request: Request, call_next):
        """Point d'entrée principal du middleware"""
        start_time = time.time()
        
        try:
            # Vérification endpoint public
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                await self._log_request(request, response, start_time)
                return response
            
            # Vérification IP bloquée
            if await self._is_ip_blocked(request):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="IP temporairement bloquée"
                )
            
            # Authentification JWT
            auth_data = await self._authenticate_request(request)
            
            # Vérification permissions
            await self._check_permissions(request, auth_data)
            
            # Limitation de taux par utilisateur
            await self._check_rate_limit(auth_data.user_id)
            
            # Ajout des données auth à la requête
            request.state.auth_data = auth_data
            request.state.user_id = auth_data.user_id
            
            # Traitement de la requête
            response = await call_next(request)
            
            # Post-traitement
            await self._update_session_activity(auth_data)
            await self._log_user_activity(request, auth_data)
            self._add_security_headers(response, auth_data)
            
            # Métriques
            await self._record_metrics(request, response, start_time, auth_data)
            
            return response
            
        except HTTPException as e:
            await self._log_auth_failure(request, str(e.detail))
            raise
        except Exception as e:
            logger.error(f"Erreur middleware auth: {str(e)}")
            await self._log_auth_failure(request, "Erreur interne")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur d'authentification"
            )

    def _is_public_endpoint(self, path: str) -> bool:
        """Vérifier si l'endpoint est public"""
        return any(path.startswith(endpoint) for endpoint in self.public_endpoints)

    async def _is_ip_blocked(self, request: Request) -> bool:
        """Vérifier si l'IP est bloquée"""
        ip_address = self._get_client_ip(request)
        block_key = f"blocked_ip:{ip_address}"
        return await self.redis_client.exists(block_key)

    async def _authenticate_request(self, request: Request) -> AuthTokenData:
        """Authentification principale via JWT"""
        try:
            # Extraction du token
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token d'authentification manquant",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            token = authorization.split(" ")[1]
            
            # Vérification token dans blacklist
            if await self._is_token_blacklisted(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token révoqué"
                )
            
            # Décodage JWT
            try:
                payload = jwt.decode(
                    token,
                    settings.JWT_SECRET_KEY,
                    algorithms=[settings.JWT_ALGORITHM]
                )
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expiré"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token invalide"
                )
            
            # Validation session Redis
            session_data = await self._validate_session(payload.get("session_id"))
            if not session_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expirée"
                )
            
            # Construction données auth
            auth_data = AuthTokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                role=payload.get("role", "user"),
                spotify_id=payload.get("spotify_id"),
                permissions=payload.get("permissions", []),
                session_id=payload["session_id"],
                device_id=payload.get("device_id"),
                ip_address=self._get_client_ip(request),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                issued_at=datetime.fromtimestamp(payload["iat"])
            )
            
            return auth_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur authentification: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Erreur d'authentification"
            )

    async def _is_token_blacklisted(self, token: str) -> bool:
        """Vérifier si le token est dans la blacklist"""
        token_hash = self.security_utils.hash_token(token)
        blacklist_key = f"blacklist_token:{token_hash}"
        return await self.redis_client.exists(blacklist_key)

    async def _validate_session(self, session_id: str) -> Optional[Dict]:
        """Valider la session dans Redis"""
        if not session_id:
            return None
        
        session_key = f"session:{session_id}"
        session_data = await self.redis_client.get(session_key)
        
        if session_data:
            return json.loads(session_data)
        return None

    async def _check_permissions(self, request: Request, auth_data: AuthTokenData):
        """Vérifier les permissions d'accès"""
        path = request.url.path
        
        # Endpoints admin
        if any(path.startswith(endpoint) for endpoint in self.admin_endpoints):
            if auth_data.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Accès administrateur requis"
                )
        
        # Permissions spécifiques
        required_permission = self._get_required_permission(request.method, path)
        if required_permission and required_permission not in auth_data.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' requise"
            )

    def _get_required_permission(self, method: str, path: str) -> Optional[str]:
        """Déterminer la permission requise pour un endpoint"""
        permission_map = {
            "GET": {
                "/api/playlists": "playlists:read",
                "/api/analytics": "analytics:read",
                "/api/recommendations": "recommendations:read"
            },
            "POST": {
                "/api/playlists": "playlists:create",
                "/api/music/analyze": "music:analyze"
            },
            "PUT": {
                "/api/playlists": "playlists:update"
            },
            "DELETE": {
                "/api/playlists": "playlists:delete"
            }
        }
        
        return permission_map.get(method, {}).get(path)

    async def _check_rate_limit(self, user_id: str):
        """Limitation de taux par utilisateur"""
        current_minute = int(time.time() // 60)
        rate_key = f"rate_limit:{user_id}:{current_minute}"
        
        current_count = await self.redis_client.get(rate_key)
        if current_count and int(current_count) >= settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Limite de taux dépassée"
            )
        
        # Incrémenter compteur
        await self.redis_client.incr(rate_key)
        await self.redis_client.expire(rate_key, 60)

    async def _update_session_activity(self, auth_data: AuthTokenData):
        """Mettre à jour l'activité de la session"""
        session_key = f"session:{auth_data.session_id}"
        session_info = await self.redis_client.get(session_key)
        
        if session_info:
            session_data = json.loads(session_info)
            session_data["last_activity"] = datetime.utcnow().isoformat()
            session_data["last_ip"] = auth_data.ip_address
            
            await self.redis_client.setex(
                session_key,
                timedelta(hours=24),
                json.dumps(session_data)
            )

    async def _log_user_activity(self, request: Request, auth_data: AuthTokenData):
        """Logger l'activité utilisateur"""
        activity_data = {
            "user_id": auth_data.user_id,
            "action": f"{request.method} {request.url.path}",
            "ip_address": auth_data.ip_address,
            "user_agent": request.headers.get("user-agent", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": auth_data.session_id
        }
        
        # Enregistrer dans Redis pour l'analyse en temps réel
        activity_key = f"activity:{auth_data.user_id}:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.redis_client.lpush(activity_key, json.dumps(activity_data))
        await self.redis_client.expire(activity_key, timedelta(days=7))

    async def _log_auth_failure(self, request: Request, error_msg: str):
        """Enregistrer les échecs d'authentification"""
        failure_data = {
            "ip_address": self._get_client_ip(request),
            "endpoint": f"{request.method} {request.url.path}",
            "user_agent": request.headers.get("user-agent", ""),
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Enregistrer pour détection d'attaques
        failure_key = f"auth_failures:{failure_data['ip_address']}"
        await self.redis_client.lpush(failure_key, json.dumps(failure_data))
        await self.redis_client.expire(failure_key, timedelta(hours=1))
        
        # Vérifier si l'IP doit être bloquée
        await self._check_and_block_suspicious_ip(failure_data["ip_address"])

    async def _check_and_block_suspicious_ip(self, ip_address: str):
        """Vérifier et bloquer les IP suspectes"""
        failure_key = f"auth_failures:{ip_address}"
        failure_count = await self.redis_client.llen(failure_key)
        
        if failure_count >= 5:  # 5 échecs en 1 heure
            # Bloquer l'IP temporairement
            block_key = f"blocked_ip:{ip_address}"
            await self.redis_client.setex(
                block_key,
                timedelta(hours=1),
                "blocked_for_suspicious_activity"
            )
            
            logger.warning(f"IP {ip_address} bloquée pour activité suspecte")

    def _add_security_headers(self, response: Response, auth_data: AuthTokenData):
        """Ajouter des headers de sécurité"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-User-ID"] = auth_data.user_id
        response.headers["X-Session-ID"] = auth_data.session_id

    async def _record_metrics(self, request: Request, response: Response, 
                            start_time: float, auth_data: AuthTokenData):
        """Enregistrer les métriques d'authentification"""
        duration = time.time() - start_time
        
        # Métriques Prometheus
        metrics.get_or_create_counter(
            "auth_requests_total",
            "Total authentication requests"
        ).labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            user_role=auth_data.role
        ).inc()
        
        # Métriques Redis pour analyse
        metrics_data = {
            "user_id": auth_data.user_id,
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        metrics_key = f"auth_metrics:{datetime.utcnow().strftime('%Y%m%d%H')}"
        await self.redis_client.lpush(metrics_key, json.dumps(metrics_data))
        await self.redis_client.expire(metrics_key, timedelta(days=1))

    async def _log_request(self, request: Request, response: Response, start_time: float):
        """Logger les requêtes publiques"""
        duration = time.time() - start_time
        logger.info(
            f"Public request: {request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s"
        )

    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'adresse IP réelle du client"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class SpotifyAuthMiddleware:
    """
    Middleware spécialisé pour l'authentification Spotify
    Gestion OAuth2 et tokens Spotify
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8", 
            decode_responses=True
        )

    async def verify_spotify_auth(self, user_id: str) -> bool:
        """Vérifier l'authentification Spotify pour un utilisateur"""
        spotify_key = f"spotify_auth:{user_id}"
        spotify_data = await self.redis_client.get(spotify_key)
        
        if not spotify_data:
            return False
        
        auth_data = json.loads(spotify_data)
        expires_at = datetime.fromisoformat(auth_data["expires_at"])
        
        # Vérifier expiration
        if expires_at <= datetime.utcnow():
            # Tenter de rafraîchir le token
            return await self._refresh_spotify_token(user_id, auth_data)
        
        return True

    async def get_spotify_token(self, user_id: str) -> Optional[str]:
        """Récupérer le token Spotify valide"""
        if not await self.verify_spotify_auth(user_id):
            return None
        
        spotify_key = f"spotify_auth:{user_id}"
        spotify_data = await self.redis_client.get(spotify_key)
        
        if spotify_data:
            auth_data = json.loads(spotify_data)
            return auth_data["access_token"]
        
        return None

    async def _refresh_spotify_token(self, user_id: str, auth_data: Dict) -> bool:
        """Rafraîchir le token Spotify"""
        try:
            # Implémentation du refresh token Spotify
            # (nécessite intégration avec l'API Spotify)
            
            # Pour l'instant, retourner False
            # TODO: Implémenter la logique de refresh
            logger.warning(f"Token Spotify expiré pour l'utilisateur {user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Erreur refresh token Spotify: {str(e)}")
            return False


class JWTAuthMiddleware:
    """
    Utilitaires pour la gestion des tokens JWT
    """
    
    @staticmethod
    def create_access_token(user_data: Dict, session_id: str = None) -> str:
        """Créer un token d'accès JWT"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        payload = {
            "user_id": user_data["id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data.get("role", "user"),
            "spotify_id": user_data.get("spotify_id"),
            "permissions": user_data.get("permissions", []),
            "session_id": session_id,
            "iat": int(time.time()),
            "exp": int(time.time() + settings.JWT_EXPIRATION_TIME),
            "device_id": user_data.get("device_id"),
            "ip_address": user_data.get("ip_address")
        }
        
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Créer un token de rafraîchissement"""
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": int(time.time()),
            "exp": int(time.time() + settings.JWT_REFRESH_EXPIRATION_TIME)
        }
        
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Dict:
        """Décoder un token JWT"""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expiré"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide"
            )


class RoleBasedAuthMiddleware:
    """
    Middleware pour l'autorisation basée sur les rôles
    """
    
    ROLE_HIERARCHY = {
        "admin": 100,
        "moderator": 50,
        "premium": 30,
        "user": 10,
        "guest": 1
    }
    
    @classmethod
    def check_role_permission(cls, user_role: str, required_role: str) -> bool:
        """Vérifier si l'utilisateur a le rôle requis"""
        user_level = cls.ROLE_HIERARCHY.get(user_role, 0)
        required_level = cls.ROLE_HIERARCHY.get(required_role, 100)
        
        return user_level >= required_level

    @classmethod
    def require_role(cls, required_role: str):
        """Décorateur pour exiger un rôle spécifique"""
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                if not hasattr(request.state, "auth_data"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentification requise"
                    )
                
                auth_data = request.state.auth_data
                if not cls.check_role_permission(auth_data.role, required_role):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Rôle '{required_role}' requis"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator


class APIKeyAuthMiddleware:
    """
    Middleware pour l'authentification par clé API
    Pour les intégrations externes et services
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def verify_api_key(self, request: Request) -> Optional[Dict]:
        """Vérifier une clé API"""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return None
        
        # Vérifier dans Redis
        api_key_data = await self.redis_client.get(f"api_key:{api_key}")
        if not api_key_data:
            return None
        
        key_info = json.loads(api_key_data)
        
        # Vérifier expiration
        if key_info.get("expires_at"):
            expires_at = datetime.fromisoformat(key_info["expires_at"])
            if expires_at <= datetime.utcnow():
                return None
        
        # Vérifier limite d'utilisation
        if key_info.get("usage_limit"):
            current_usage = key_info.get("current_usage", 0)
            if current_usage >= key_info["usage_limit"]:
                return None
        
        # Incrémenter l'utilisation
        key_info["current_usage"] = key_info.get("current_usage", 0) + 1
        key_info["last_used"] = datetime.utcnow().isoformat()
        
        await self.redis_client.set(
            f"api_key:{api_key}",
            json.dumps(key_info)
        )
        
        return key_info

    async def create_api_key(self, client_name: str, permissions: List[str], 
                           expires_days: int = 365) -> str:
        """Créer une nouvelle clé API"""
        api_key = f"sk-{str(uuid.uuid4()).replace('-', '')}"
        
        key_info = {
            "client_name": client_name,
            "permissions": permissions,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat(),
            "current_usage": 0,
            "is_active": True
        }
        
        await self.redis_client.set(
            f"api_key:{api_key}",
            json.dumps(key_info)
        )
        
        return api_key
