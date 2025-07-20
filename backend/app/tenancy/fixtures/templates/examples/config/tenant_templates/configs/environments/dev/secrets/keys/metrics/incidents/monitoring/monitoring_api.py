# =============================================================================
# Monitoring API Enterprise - FastAPI Ultra-Avancé
# =============================================================================
# 
# API REST enterprise pour gestion et consultation du système de monitoring
# avec authentification, autorisation, rate limiting et documentation auto.
#
# Architecture moderne:
# - FastAPI avec validation Pydantic avancée
# - Authentification JWT et RBAC
# - Rate limiting et audit logging
# - Documentation OpenAPI/Swagger complète
# - Middleware de sécurité enterprise
# - Support multi-tenant avec isolation
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture API enterprise)
# - Backend Senior Developer (Python/FastAPI/Django)
# - Spécialiste Sécurité Backend (Auth, RBAC, audit)
# - Architecte Microservices (API design et patterns)
# - DBA & Data Engineer (Optimisation queries)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Annotated
import uuid
import json
from pathlib import Path

# FastAPI et dépendances
from fastapi import (
    FastAPI, HTTPException, Depends, Security, Request, Response,
    BackgroundTasks, Query, Path as PathParam, Body, Header
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

# Validation et modèles
from pydantic import BaseModel, Field, validator, EmailStr, root_validator
from pydantic.dataclasses import dataclass
from enum import Enum

# Sécurité et authentification
import jwt
from passlib.context import CryptContext
from jose import JWTError
import bcrypt

# Monitoring et observabilité
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import opentelemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Imports asyncio et bases de données
import aioredis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Rate limiting et cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiocache

# Imports locaux
from config_manager import ConfigurationManager, setup_monitoring_config
from __init__ import (
    EnterpriseMonitoringOrchestrator,
    MultiTenantMonitoringManager,
    MonitoringFactory,
    initialize_monitoring,
    MonitoringConfig,
    MonitoringHealth,
    MonitoringTier
)

# Configuration logging structuré
logger = structlog.get_logger(__name__)

# =============================================================================
# CONFIGURATION DE L'API
# =============================================================================

# Configuration sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Cache distribué
cache = aiocache.Cache(aiocache.SimpleMemoryCache)

# Métriques Prometheus
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code', 'tenant_id']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint', 'tenant_id']
)

active_users = Gauge(
    'api_active_users',
    'Number of active users',
    ['tenant_id']
)

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class UserRole(str, Enum):
    """Rôles utilisateur"""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class APIResponse(BaseModel):
    """Réponse API standardisée"""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class UserModel(BaseModel):
    """Modèle utilisateur"""
    id: Optional[str] = None
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.VIEWER
    tenant_id: str = Field(..., min_length=1)
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)

class UserCreate(BaseModel):
    """Modèle création utilisateur"""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.VIEWER
    tenant_id: str = Field(..., min_length=1)

class LoginRequest(BaseModel):
    """Requête de connexion"""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    tenant_id: Optional[str] = None

class TokenData(BaseModel):
    """Données du token JWT"""
    user_id: str
    username: str
    tenant_id: str
    role: UserRole
    permissions: List[str] = Field(default_factory=list)
    exp: datetime

class IncidentCreate(BaseModel):
    """Modèle création incident"""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    severity: AlertSeverity
    category: str = Field(..., min_length=1, max_length=50)
    source: str = Field(default="api", max_length=50)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IncidentUpdate(BaseModel):
    """Modèle mise à jour incident"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=2000)
    severity: Optional[AlertSeverity] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class IncidentFilter(BaseModel):
    """Filtres pour recherche d'incidents"""
    severity: Optional[AlertSeverity] = None
    category: Optional[str] = None
    status: Optional[str] = None
    source: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    search: Optional[str] = None

class MetricsQuery(BaseModel):
    """Requête de métriques"""
    metrics: List[str] = Field(..., min_items=1)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step: Optional[str] = Field(None, regex=r'^\d+[smhd]$')
    filters: Dict[str, str] = Field(default_factory=dict)

class DashboardCreate(BaseModel):
    """Modèle création dashboard"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config: Dict[str, Any] = Field(...)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = False

class AlertRuleCreate(BaseModel):
    """Modèle création règle d'alerte"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    query: str = Field(..., min_length=1)
    for_duration: str = Field(default="5m", regex=r'^\d+[smh]$')
    severity: AlertSeverity = AlertSeverity.WARNING
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

# =============================================================================
# GESTION DE L'AUTHENTIFICATION
# =============================================================================

class AuthManager:
    """Gestionnaire d'authentification et autorisation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jwt_secret = config.get('jwt_secret', 'monitoring-secret-2025')
        self.jwt_expiration_hours = config.get('jwt_expiration_hours', 24)
        self.algorithm = "HS256"
        
        # Base d'utilisateurs (en production, utiliser une vraie DB)
        self.users_db: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        self._create_default_users()

    def _create_default_users(self):
        """Création des utilisateurs par défaut"""
        default_users = [
            {
                'username': 'admin',
                'email': 'admin@monitoring.local',
                'password': 'monitoring_admin_2025!',
                'full_name': 'Administrator',
                'role': UserRole.SUPER_ADMIN,
                'tenant_id': 'system',
                'permissions': ['*']
            },
            {
                'username': 'operator',
                'email': 'operator@monitoring.local',
                'password': 'monitoring_op_2025!',
                'full_name': 'Operator',
                'role': UserRole.OPERATOR,
                'tenant_id': 'default',
                'permissions': ['read', 'write', 'alert']
            },
            {
                'username': 'viewer',
                'email': 'viewer@monitoring.local',
                'password': 'monitoring_view_2025!',
                'full_name': 'Viewer',
                'role': UserRole.VIEWER,
                'tenant_id': 'default',
                'permissions': ['read']
            }
        ]
        
        for user_data in default_users:
            user_id = str(uuid.uuid4())
            hashed_password = pwd_context.hash(user_data['password'])
            
            self.users_db[user_id] = {
                'id': user_id,
                'username': user_data['username'],
                'email': user_data['email'],
                'password_hash': hashed_password,
                'full_name': user_data['full_name'],
                'role': user_data['role'],
                'tenant_id': user_data['tenant_id'],
                'permissions': user_data['permissions'],
                'is_active': True,
                'created_at': datetime.utcnow(),
                'last_login': None
            }

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Vérification du mot de passe"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hashage du mot de passe"""
        return pwd_context.hash(password)

    def authenticate_user(self, username: str, password: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Authentification utilisateur"""
        
        for user_id, user in self.users_db.items():
            if (user['username'] == username and 
                user['is_active'] and
                (tenant_id is None or user['tenant_id'] == tenant_id)):
                
                if self.verify_password(password, user['password_hash']):
                    # Mise à jour de la dernière connexion
                    user['last_login'] = datetime.utcnow()
                    return user
        
        return None

    def create_access_token(self, user: Dict[str, Any]) -> str:
        """Création d'un token d'accès JWT"""
        
        expire = datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours)
        
        token_data = {
            'user_id': user['id'],
            'username': user['username'],
            'tenant_id': user['tenant_id'],
            'role': user['role'].value if isinstance(user['role'], UserRole) else user['role'],
            'permissions': user['permissions'],
            'exp': expire,
            'iat': datetime.utcnow(),
            'iss': 'monitoring-api'
        }
        
        token = jwt.encode(token_data, self.jwt_secret, algorithm=self.algorithm)
        
        # Enregistrement de la session
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user['id'],
            'token': token,
            'created_at': datetime.utcnow(),
            'expires_at': expire,
            'last_activity': datetime.utcnow()
        }
        
        return token

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Vérification d'un token JWT"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            
            # Vérification de l'expiration
            exp = datetime.fromtimestamp(payload.get('exp', 0))
            if datetime.utcnow() > exp:
                return None
            
            return TokenData(
                user_id=payload.get('user_id'),
                username=payload.get('username'),
                tenant_id=payload.get('tenant_id'),
                role=UserRole(payload.get('role')),
                permissions=payload.get('permissions', []),
                exp=exp
            )
            
        except JWTError:
            return None

    def has_permission(self, user: TokenData, required_permission: str) -> bool:
        """Vérification des permissions"""
        
        # Super admin a tous les droits
        if user.role == UserRole.SUPER_ADMIN:
            return True
        
        # Vérification permission wildcard
        if '*' in user.permissions:
            return True
        
        # Vérification permission exacte
        if required_permission in user.permissions:
            return True
        
        # Vérification permissions par rôle
        role_permissions = {
            UserRole.VIEWER: ['read'],
            UserRole.OPERATOR: ['read', 'write', 'alert'],
            UserRole.ADMIN: ['read', 'write', 'alert', 'admin'],
            UserRole.SUPER_ADMIN: ['*']
        }
        
        user_role_permissions = role_permissions.get(user.role, [])
        return required_permission in user_role_permissions

# =============================================================================
# DÉPENDANCES FASTAPI
# =============================================================================

# Instance globale des managers
auth_manager: Optional[AuthManager] = None
monitoring_orchestrator: Optional[EnterpriseMonitoringOrchestrator] = None
config_manager: Optional[ConfigurationManager] = None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Récupération de l'utilisateur actuel"""
    
    global auth_manager
    
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Service d'authentification non disponible")
    
    token_data = auth_manager.verify_token(credentials.credentials)
    
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_data

def require_permission(permission: str):
    """Décorateur pour vérifier les permissions"""
    
    def permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        global auth_manager
        
        if not auth_manager.has_permission(current_user, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission requise: {permission}"
            )
        
        return current_user
    
    return permission_checker

async def get_monitoring_orchestrator() -> EnterpriseMonitoringOrchestrator:
    """Récupération de l'orchestrateur de monitoring"""
    
    global monitoring_orchestrator
    
    if not monitoring_orchestrator:
        raise HTTPException(status_code=500, detail="Service de monitoring non disponible")
    
    return monitoring_orchestrator

# =============================================================================
# MIDDLEWARE PERSONNALISÉS
# =============================================================================

class SecurityHeadersMiddleware:
    """Middleware pour les en-têtes de sécurité"""
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # En-têtes de sécurité
                    security_headers = {
                        b"X-Content-Type-Options": b"nosniff",
                        b"X-Frame-Options": b"DENY",
                        b"X-XSS-Protection": b"1; mode=block",
                        b"Strict-Transport-Security": b"max-age=31536000; includeSubDomains",
                        b"Content-Security-Policy": b"default-src 'self'",
                        b"Referrer-Policy": b"strict-origin-when-cross-origin"
                    }
                    
                    headers.update(security_headers)
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

class RequestLoggingMiddleware:
    """Middleware pour logging des requêtes"""
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Logging de la requête entrante
            logger.info("Request started", 
                       request_id=request_id,
                       method=scope["method"],
                       path=scope["path"],
                       query_string=scope.get("query_string", b"").decode())
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Ajout de l'ID de requête dans les en-têtes
                    headers = dict(message.get("headers", []))
                    headers[b"X-Request-ID"] = request_id.encode()
                    message["headers"] = list(headers.items())
                    
                    # Logging de la réponse
                    duration = time.time() - start_time
                    status_code = message["status"]
                    
                    logger.info("Request completed",
                               request_id=request_id,
                               status_code=status_code,
                               duration_ms=round(duration * 1000, 2))
                    
                    # Métriques Prometheus
                    api_requests_total.labels(
                        method=scope["method"],
                        endpoint=scope["path"],
                        status_code=status_code,
                        tenant_id="unknown"
                    ).inc()
                    
                    api_request_duration.labels(
                        method=scope["method"],
                        endpoint=scope["path"],
                        tenant_id="unknown"
                    ).observe(duration)
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# =============================================================================
# CRÉATION DE L'APPLICATION FASTAPI
# =============================================================================

def create_monitoring_api() -> FastAPI:
    """Création de l'application FastAPI"""
    
    app = FastAPI(
        title="Monitoring API Enterprise",
        description="""
        API REST enterprise pour le système de monitoring avancé.
        
        ## Fonctionnalités
        
        * **Authentification JWT** avec RBAC
        * **Gestion des incidents** avec workflow complet
        * **Métriques et dashboards** avec Prometheus/Grafana
        * **Alerting intelligent** avec corrélation
        * **Multi-tenant** avec isolation complète
        * **Audit logging** et traçabilité
        * **Rate limiting** et sécurité avancée
        
        ## Architecture
        
        Développé par l'équipe d'experts Achiri:
        - Lead Developer + AI Architect
        - Backend Senior Developer (Python/FastAPI)
        - Spécialiste Sécurité Backend
        - Architecte Microservices
        
        Direction Technique: **Fahed Mlaiel**
        """,
        version="1.0.0",
        contact={
            "name": "Équipe Monitoring Achiri",
            "email": "monitoring@achiri.com",
        },
        license_info={
            "name": "Propriétaire Achiri",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En production, spécifier les domaines autorisés
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de sécurité
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    return app

# Instance de l'application
app = create_monitoring_api()

# =============================================================================
# ROUTES D'AUTHENTIFICATION
# =============================================================================

@app.post("/auth/login", response_model=APIResponse, tags=["Authentication"])
@limiter.limit("10/minute")
async def login(request: Request, login_data: LoginRequest):
    """Connexion utilisateur"""
    
    global auth_manager
    
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Service d'authentification non disponible")
    
    user = auth_manager.authenticate_user(
        login_data.username, 
        login_data.password, 
        login_data.tenant_id
    )
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Identifiants invalides"
        )
    
    token = auth_manager.create_access_token(user)
    
    return APIResponse(
        success=True,
        message="Connexion réussie",
        data={
            "access_token": token,
            "token_type": "bearer",
            "expires_in": auth_manager.jwt_expiration_hours * 3600,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "tenant_id": user["tenant_id"]
            }
        }
    )

@app.post("/auth/logout", response_model=APIResponse, tags=["Authentication"])
async def logout(current_user: TokenData = Depends(get_current_user)):
    """Déconnexion utilisateur"""
    
    # En production, invalider le token côté serveur
    
    return APIResponse(
        success=True,
        message="Déconnexion réussie"
    )

@app.get("/auth/me", response_model=APIResponse, tags=["Authentication"])
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Informations de l'utilisateur actuel"""
    
    return APIResponse(
        success=True,
        data={
            "user_id": current_user.user_id,
            "username": current_user.username,
            "tenant_id": current_user.tenant_id,
            "role": current_user.role,
            "permissions": current_user.permissions,
            "token_expires": current_user.exp.isoformat()
        }
    )

# =============================================================================
# ROUTES DE MONITORING
# =============================================================================

@app.get("/health", response_model=APIResponse, tags=["System"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Vérification de santé de l'API"""
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": int(time.time() - app.state.start_time) if hasattr(app.state, 'start_time') else 0
    }
    
    return APIResponse(
        success=True,
        message="API opérationnelle",
        data=health_data
    )

@app.get("/system/status", response_model=APIResponse, tags=["System"])
async def get_system_status(
    current_user: TokenData = Depends(require_permission("read")),
    orchestrator: EnterpriseMonitoringOrchestrator = Depends(get_monitoring_orchestrator)
):
    """Statut du système de monitoring"""
    
    health = await orchestrator.get_system_health()
    metrics_summary = await orchestrator.get_metrics_summary(current_user.tenant_id)
    
    return APIResponse(
        success=True,
        data={
            "health": {
                "overall_status": health.overall_status.value,
                "components": {k: v.value for k, v in health.components.items()},
                "last_check": health.last_check.isoformat(),
                "alerts_active": health.alerts_active,
                "errors": health.errors
            },
            "metrics": metrics_summary
        }
    )

@app.post("/incidents", response_model=APIResponse, tags=["Incidents"])
async def create_incident(
    incident_data: IncidentCreate,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("write")),
    orchestrator: EnterpriseMonitoringOrchestrator = Depends(get_monitoring_orchestrator)
):
    """Création d'un incident"""
    
    # Enregistrement de l'incident
    await orchestrator.record_incident(
        tenant_id=current_user.tenant_id,
        severity=incident_data.severity.value,
        category=incident_data.category,
        source=incident_data.source,
        metadata={
            "title": incident_data.title,
            "description": incident_data.description,
            "tags": incident_data.tags,
            "created_by": current_user.username,
            **incident_data.metadata
        }
    )
    
    incident_id = str(uuid.uuid4())
    
    return APIResponse(
        success=True,
        message="Incident créé avec succès",
        data={
            "incident_id": incident_id,
            "tenant_id": current_user.tenant_id,
            "created_at": datetime.utcnow().isoformat()
        }
    )

@app.get("/metrics/query", response_model=APIResponse, tags=["Metrics"])
@limiter.limit("50/minute")
async def query_metrics(
    request: Request,
    query: str = Query(..., description="Requête PromQL"),
    start: Optional[datetime] = Query(None, description="Début de la période"),
    end: Optional[datetime] = Query(None, description="Fin de la période"),
    step: Optional[str] = Query("1m", regex=r'^\d+[smhd]$', description="Pas de temps"),
    current_user: TokenData = Depends(require_permission("read"))
):
    """Requête de métriques Prometheus"""
    
    # En production, proxy vers Prometheus avec filtrage par tenant
    mock_data = {
        "query": query,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "step": step,
        "tenant_id": current_user.tenant_id,
        "results": [
            {
                "metric": {"__name__": "api_requests_total", "job": "monitoring-api"},
                "values": [[time.time(), "42"]]
            }
        ]
    }
    
    return APIResponse(
        success=True,
        data=mock_data
    )

@app.get("/dashboards", response_model=APIResponse, tags=["Dashboards"])
async def list_dashboards(
    current_user: TokenData = Depends(require_permission("read"))
):
    """Liste des dashboards disponibles"""
    
    # En production, récupérer depuis la base de données
    mock_dashboards = [
        {
            "id": "system-overview",
            "name": "Vue d'ensemble système",
            "description": "Métriques système générales",
            "tenant_id": current_user.tenant_id,
            "created_at": datetime.utcnow().isoformat()
        },
        {
            "id": "api-metrics",
            "name": "Métriques API",
            "description": "Performance et utilisation de l'API",
            "tenant_id": current_user.tenant_id,
            "created_at": datetime.utcnow().isoformat()
        }
    ]
    
    return APIResponse(
        success=True,
        data={"dashboards": mock_dashboards}
    )

@app.post("/dashboards", response_model=APIResponse, tags=["Dashboards"])
async def create_dashboard(
    dashboard_data: DashboardCreate,
    current_user: TokenData = Depends(require_permission("write")),
    orchestrator: EnterpriseMonitoringOrchestrator = Depends(get_monitoring_orchestrator)
):
    """Création d'un dashboard personnalisé"""
    
    dashboard_id = await orchestrator.create_custom_dashboard(
        name=dashboard_data.name,
        tenant_id=current_user.tenant_id,
        config={
            "description": dashboard_data.description,
            "config": dashboard_data.config,
            "tags": dashboard_data.tags,
            "is_public": dashboard_data.is_public,
            "created_by": current_user.username
        }
    )
    
    return APIResponse(
        success=True,
        message="Dashboard créé avec succès",
        data={
            "dashboard_id": dashboard_id,
            "tenant_id": current_user.tenant_id
        }
    )

# =============================================================================
# ROUTES D'ADMINISTRATION
# =============================================================================

@app.get("/admin/config", response_model=APIResponse, tags=["Administration"])
async def get_configuration(
    component: Optional[str] = Query(None, description="Composant spécifique"),
    current_user: TokenData = Depends(require_permission("admin"))
):
    """Récupération de la configuration"""
    
    global config_manager
    
    if not config_manager:
        raise HTTPException(status_code=500, detail="Gestionnaire de configuration non disponible")
    
    if component:
        config_data = config_manager.load_config(component)
        return APIResponse(
            success=True,
            data={component: config_data}
        )
    else:
        config_summary = config_manager.get_config_summary()
        return APIResponse(
            success=True,
            data=config_summary
        )

@app.get("/admin/metrics/prometheus", tags=["Administration"])
async def prometheus_metrics(
    current_user: TokenData = Depends(require_permission("read"))
):
    """Export des métriques Prometheus"""
    
    # Generation des métriques Prometheus
    metrics_data = generate_latest()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

# =============================================================================
# GESTION DES ÉVÉNEMENTS DE L'APPLICATION
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'application"""
    
    global auth_manager, monitoring_orchestrator, config_manager
    
    try:
        app.state.start_time = time.time()
        
        # Initialisation du gestionnaire de configuration
        config_manager = setup_monitoring_config("dev")
        
        # Initialisation du gestionnaire d'authentification
        security_config = config_manager.load_config("security")
        auth_manager = AuthManager(security_config)
        
        # Initialisation du monitoring
        monitoring_config = MonitoringFactory.create_default_config()
        monitoring_orchestrator = await initialize_monitoring(monitoring_config)
        
        # Instrumentation OpenTelemetry
        FastAPIInstrumentor.instrument_app(app)
        
        logger.info("API de monitoring démarrée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur démarrage API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt de l'application"""
    
    global monitoring_orchestrator
    
    try:
        if monitoring_orchestrator:
            await monitoring_orchestrator.shutdown()
        
        logger.info("API de monitoring arrêtée")
        
    except Exception as e:
        logger.error(f"Erreur arrêt API: {e}")

# =============================================================================
# POINT D'ENTRÉE POUR DÉVELOPPEMENT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "monitoring_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
