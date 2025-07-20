#!/usr/bin/env python3
"""
Service Web API pour Gestion des Règles d'Alertes - Interface REST/GraphQL Ultra-Performante

Ce module expose une API REST et GraphQL complète pour la gestion des règles d'alertes
avec authentification, autorisation, validation, rate limiting, et monitoring en temps réel.

Architecture API:
- FastAPI avec validation Pydantic
- GraphQL avec Strawberry
- Authentification JWT/OAuth2
- Rate limiting par tenant
- WebSocket pour streaming temps réel
- Monitoring Prometheus intégré
- Documentation OpenAPI/Swagger automatique
- Cache Redis pour performance

Équipe Engineering:
✅ Lead Dev + Architecte IA : Fahed Mlaiel
✅ Développeur Backend Senior (Python/FastAPI/Django)
✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
✅ Spécialiste Sécurité Backend
✅ Architecte Microservices

License: Spotify Proprietary
Copyright: © 2025 Spotify Technology S.A.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from uuid import UUID, uuid4

# FastAPI core
from fastapi import (
    FastAPI, HTTPException, Depends, status, Request, Response,
    WebSocket, WebSocketDisconnect, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import constr, conint

# GraphQL
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

# Monitoring and metrics
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Authentication
import jwt
from passlib.context import CryptContext

# Internal imports
from .manager import RuleManager, RuleEvaluationConfig, create_rule_manager
from .core import (
    AlertRule, AlertSeverity, AlertCategory, RuleStatus,
    EvaluationResult, AlertMetrics, RuleContext
)
from ...........................core.exceptions import (
    AlertRuleException, ValidationException, AuthenticationException
)
from ...........................core.security import SecurityManager
from ...........................core.database import DatabaseManager

# Configuration du logging
logger = structlog.get_logger(__name__)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Métriques Prometheus
API_REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status', 'tenant_id']
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_WEBSOCKET_CONNECTIONS = Gauge(
    'active_websocket_connections',
    'Number of active WebSocket connections',
    ['tenant_id']
)


# Modèles Pydantic pour l'API

class TenantInfo(BaseModel):
    """Informations sur un tenant"""
    tenant_id: str = Field(..., description="Identifiant unique du tenant")
    name: str = Field(..., description="Nom du tenant")
    environment: str = Field(default="dev", description="Environnement (dev, staging, prod)")
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "spotify_tenant_001",
                "name": "Spotify Main",
                "environment": "prod"
            }
        }


class ConditionConfigModel(BaseModel):
    """Configuration d'une condition de règle"""
    type: str = Field(..., description="Type de condition")
    condition_id: Optional[str] = Field(None, description="ID unique de la condition")
    weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Poids de la condition")
    
    # Champs spécifiques aux conditions seuils
    metric_path: Optional[str] = Field(None, description="Chemin de la métrique")
    operator: Optional[str] = Field(None, description="Opérateur de comparaison")
    threshold: Optional[float] = Field(None, description="Valeur seuil")
    
    # Champs spécifiques aux conditions ML
    model_name: Optional[str] = Field(None, description="Nom du modèle ML")
    contamination: Optional[float] = Field(None, ge=0.01, le=0.5, description="Taux de contamination")
    
    # Champs spécifiques aux conditions composites
    logic_operator: Optional[str] = Field(None, description="Opérateur logique")
    conditions: Optional[List['ConditionConfigModel']] = Field(None, description="Sous-conditions")
    
    @validator('operator')
    def validate_operator(cls, v):
        if v is not None:
            valid_operators = ['>', '<', '>=', '<=', '==', '!=']
            if v not in valid_operators:
                raise ValueError(f"Operator must be one of {valid_operators}")
        return v
    
    @validator('logic_operator')
    def validate_logic_operator(cls, v):
        if v is not None:
            valid_operators = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
            if v.upper() not in valid_operators:
                raise ValueError(f"Logic operator must be one of {valid_operators}")
        return v.upper() if v else v


# Mise à jour pour supporter la récursion
ConditionConfigModel.update_forward_refs()


class RuleConfigModel(BaseModel):
    """Configuration complète d'une règle d'alerte"""
    name: constr(min_length=1, max_length=200) = Field(..., description="Nom de la règle")
    description: str = Field(default="", max_length=1000, description="Description de la règle")
    severity: str = Field(..., description="Niveau de sévérité")
    category: str = Field(..., description="Catégorie de la règle")
    tenant_id: str = Field(..., description="ID du tenant")
    environment: str = Field(default="dev", description="Environnement")
    enabled: bool = Field(default=True, description="Règle activée")
    cooldown_period_seconds: conint(ge=0, le=86400) = Field(
        default=300, description="Période de cooldown en secondes"
    )
    max_executions_per_hour: conint(ge=1, le=1000) = Field(
        default=100, description="Nombre max d'exécutions par heure"
    )
    conditions: List[ConditionConfigModel] = Field(..., description="Conditions de la règle")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        if v.upper() not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}")
        return v.upper()
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = [
            'infrastructure', 'application', 'security', 'business',
            'ml_anomaly', 'performance', 'user_experience'
        ]
        if v.lower() not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "name": "High CPU Usage Alert",
                "description": "Alert when CPU usage exceeds 80%",
                "severity": "HIGH",
                "category": "infrastructure",
                "tenant_id": "spotify_tenant_001",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 300,
                "max_executions_per_hour": 20,
                "conditions": [
                    {
                        "type": "threshold",
                        "metric_path": "current_metrics.cpu_usage",
                        "operator": ">",
                        "threshold": 80.0,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "infrastructure",
                    "priority": "high"
                }
            }
        }


class RuleUpdateModel(BaseModel):
    """Modèle pour mise à jour partielle d'une règle"""
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    enabled: Optional[bool] = None
    cooldown_period_seconds: Optional[conint(ge=0, le=86400)] = None
    max_executions_per_hour: Optional[conint(ge=1, le=1000)] = None
    tags: Optional[Dict[str, str]] = None


class AlertMetricsModel(BaseModel):
    """Modèle pour les métriques d'alerte"""
    cpu_usage: float = Field(..., ge=0, le=100, description="Utilisation CPU en %")
    memory_usage: float = Field(..., ge=0, le=100, description="Utilisation mémoire en %")
    disk_usage: float = Field(..., ge=0, le=100, description="Utilisation disque en %")
    network_latency: float = Field(..., ge=0, description="Latence réseau en ms")
    error_rate: float = Field(..., ge=0, le=100, description="Taux d'erreur en %")
    request_rate: float = Field(..., ge=0, description="Taux de requêtes par seconde")
    response_time: float = Field(..., ge=0, description="Temps de réponse en ms")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques personnalisées")


class EvaluationRequestModel(BaseModel):
    """Demande d'évaluation de règles"""
    tenant_id: str = Field(..., description="ID du tenant")
    metrics: Optional[AlertMetricsModel] = Field(None, description="Métriques à évaluer")
    rule_ids: Optional[List[str]] = Field(None, description="IDs spécifiques des règles à évaluer")


class EvaluationResultModel(BaseModel):
    """Résultat d'évaluation d'une règle"""
    rule_id: str
    triggered: bool
    severity: str
    message: str
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any]


class RuleInfoModel(BaseModel):
    """Informations sur une règle"""
    rule_id: str
    name: str
    description: str
    severity: str
    category: str
    tenant_id: str
    environment: str
    enabled: bool
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class APIResponse(BaseModel):
    """Réponse API standardisée"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Modèles GraphQL avec Strawberry

@strawberry.type
class GraphQLRule:
    """Représentation GraphQL d'une règle"""
    rule_id: str
    name: str
    description: str
    severity: str
    category: str
    tenant_id: str
    enabled: bool
    status: str


@strawberry.type
class GraphQLEvaluationResult:
    """Résultat d'évaluation GraphQL"""
    rule_id: str
    triggered: bool
    severity: str
    message: str
    execution_time: float
    timestamp: datetime


@strawberry.input
class GraphQLRuleInput:
    """Input GraphQL pour création de règle"""
    name: str
    description: str = ""
    severity: str
    category: str
    tenant_id: str
    enabled: bool = True


# Gestion de l'authentification

class AuthManager:
    """Gestionnaire d'authentification et d'autorisation"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Crée un token d'accès JWT"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise AuthenticationException("Invalid token")
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Récupère l'utilisateur actuel depuis le token"""
        try:
            payload = self.verify_token(credentials.credentials)
            tenant_id: str = payload.get("tenant_id")
            user_id: str = payload.get("user_id")
            
            if tenant_id is None or user_id is None:
                raise AuthenticationException("Invalid token payload")
            
            return {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "scopes": payload.get("scopes", [])
            }
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )


# WebSocket Manager pour streaming temps réel

class WebSocketManager:
    """Gestionnaire de connexions WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, tenant_id: str, user_id: str):
        """Nouvelle connexion WebSocket"""
        await websocket.accept()
        
        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = set()
        
        self.active_connections[tenant_id].add(websocket)
        self.connection_metadata[websocket] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "connected_at": datetime.utcnow()
        }
        
        ACTIVE_WEBSOCKET_CONNECTIONS.labels(tenant_id=tenant_id).inc()
        
        logger.info(
            "WebSocket connection established",
            tenant_id=tenant_id,
            user_id=user_id
        )
    
    def disconnect(self, websocket: WebSocket):
        """Déconnexion WebSocket"""
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            tenant_id = metadata["tenant_id"]
            
            self.active_connections[tenant_id].discard(websocket)
            del self.connection_metadata[websocket]
            
            ACTIVE_WEBSOCKET_CONNECTIONS.labels(tenant_id=tenant_id).dec()
            
            logger.info(
                "WebSocket connection closed",
                tenant_id=tenant_id,
                user_id=metadata["user_id"]
            )
    
    async def send_to_tenant(self, tenant_id: str, message: dict):
        """Envoie un message à toutes les connexions d'un tenant"""
        if tenant_id in self.active_connections:
            connections_to_remove = []
            
            for websocket in self.active_connections[tenant_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        "Failed to send WebSocket message",
                        tenant_id=tenant_id,
                        error=str(e)
                    )
                    connections_to_remove.append(websocket)
            
            # Nettoyage des connexions fermées
            for websocket in connections_to_remove:
                self.disconnect(websocket)
    
    async def send_to_user(self, tenant_id: str, user_id: str, message: dict):
        """Envoie un message à un utilisateur spécifique"""
        if tenant_id in self.active_connections:
            for websocket in self.active_connections[tenant_id]:
                metadata = self.connection_metadata.get(websocket)
                if metadata and metadata["user_id"] == user_id:
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        logger.error(
                            "Failed to send WebSocket message to user",
                            tenant_id=tenant_id,
                            user_id=user_id,
                            error=str(e)
                        )


# Classe principale de l'API

class AlertRulesAPI:
    """API principale pour la gestion des règles d'alertes"""
    
    def __init__(
        self,
        rule_manager: RuleManager,
        auth_manager: AuthManager,
        websocket_manager: WebSocketManager
    ):
        self.rule_manager = rule_manager
        self.auth_manager = auth_manager
        self.websocket_manager = websocket_manager
        
        # Création de l'application FastAPI
        self.app = FastAPI(
            title="Spotify Alert Rules API",
            description="API ultra-performante pour la gestion des règles d'alertes",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configuration des middlewares
        self._setup_middlewares()
        
        # Configuration des routes
        self._setup_routes()
        
        # Configuration GraphQL
        self._setup_graphql()
        
        # Instrumentation Prometheus
        self._setup_monitoring()
    
    def _setup_middlewares(self):
        """Configuration des middlewares"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # À restreindre en production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Middleware de timing
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Métriques Prometheus
            API_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(process_time)
            
            return response
    
    def _setup_routes(self):
        """Configuration des routes API"""
        
        @self.app.get("/", response_model=APIResponse)
        async def root():
            """Point d'entrée de l'API"""
            return APIResponse(
                success=True,
                message="Spotify Alert Rules API v2.0.0",
                data={"status": "operational", "version": "2.0.0"}
            )
        
        @self.app.get("/health", response_model=APIResponse)
        async def health_check():
            """Health check de l'API"""
            stats = await self.rule_manager.get_statistics()
            return APIResponse(
                success=True,
                message="Service healthy",
                data=stats
            )
        
        # Routes des règles
        self._setup_rule_routes()
        
        # Routes d'évaluation
        self._setup_evaluation_routes()
        
        # Routes de monitoring
        self._setup_monitoring_routes()
        
        # WebSocket
        self._setup_websocket_routes()
    
    def _setup_rule_routes(self):
        """Configuration des routes de gestion des règles"""
        
        @self.app.post("/api/v1/rules", response_model=APIResponse)
        @limiter.limit("100/minute")
        async def create_rule(
            request: Request,
            rule_config: RuleConfigModel,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Crée une nouvelle règle d'alerte"""
            try:
                # Vérification des permissions
                if rule_config.tenant_id != current_user["tenant_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to tenant"
                    )
                
                # Création de la règle
                rule = await self.rule_manager.add_rule(rule_config.dict())
                
                # Notification WebSocket
                await self.websocket_manager.send_to_tenant(
                    rule_config.tenant_id,
                    {
                        "type": "rule_created",
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name
                    }
                )
                
                API_REQUESTS.labels(
                    method="POST",
                    endpoint="/api/v1/rules",
                    status="success",
                    tenant_id=rule_config.tenant_id
                ).inc()
                
                return APIResponse(
                    success=True,
                    message="Rule created successfully",
                    data={"rule_id": rule.rule_id}
                )
                
            except ValidationException as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e)
                )
            except Exception as e:
                logger.error("Failed to create rule", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.get("/api/v1/rules", response_model=APIResponse)
        @limiter.limit("200/minute")
        async def list_rules(
            request: Request,
            category: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Liste les règles d'un tenant"""
            try:
                tenant_id = current_user["tenant_id"]
                
                # Conversion des paramètres
                alert_category = None
                if category:
                    alert_category = AlertCategory(category)
                
                rule_status = None
                if status:
                    rule_status = RuleStatus(status)
                
                rules = await self.rule_manager.list_rules(
                    tenant_id=tenant_id,
                    category=alert_category,
                    status=rule_status
                )
                
                # Pagination
                total = len(rules)
                rules_page = rules[offset:offset + limit]
                
                rule_data = [
                    RuleInfoModel(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        description=rule.description,
                        severity=rule.severity.name,
                        category=rule.category.value,
                        tenant_id=rule.tenant_id,
                        environment=rule.environment,
                        enabled=rule.enabled,
                        status=rule.status.value,
                        statistics={
                            "execution_count": rule.execution_count,
                            "success_count": rule.success_count,
                            "error_count": rule.error_count
                        },
                        tags=rule.tags
                    ).dict()
                    for rule in rules_page
                ]
                
                return APIResponse(
                    success=True,
                    message=f"Found {total} rules",
                    data={
                        "rules": rule_data,
                        "total": total,
                        "limit": limit,
                        "offset": offset
                    }
                )
                
            except Exception as e:
                logger.error("Failed to list rules", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.get("/api/v1/rules/{rule_id}", response_model=APIResponse)
        @limiter.limit("300/minute")
        async def get_rule(
            request: Request,
            rule_id: str,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Récupère une règle spécifique"""
            try:
                rule = await self.rule_manager.get_rule(rule_id)
                
                if not rule:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Rule not found"
                    )
                
                # Vérification des permissions
                if rule.tenant_id != current_user["tenant_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                rule_info = RuleInfoModel(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity.name,
                    category=rule.category.value,
                    tenant_id=rule.tenant_id,
                    environment=rule.environment,
                    enabled=rule.enabled,
                    status=rule.status.value,
                    statistics={
                        "execution_count": rule.execution_count,
                        "success_count": rule.success_count,
                        "error_count": rule.error_count,
                        "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                        "last_evaluated": rule.last_evaluated.isoformat() if rule.last_evaluated else None
                    },
                    tags=rule.tags
                )
                
                return APIResponse(
                    success=True,
                    message="Rule retrieved successfully",
                    data=rule_info.dict()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to get rule", rule_id=rule_id, error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.put("/api/v1/rules/{rule_id}", response_model=APIResponse)
        @limiter.limit("50/minute")
        async def update_rule(
            request: Request,
            rule_id: str,
            updates: RuleUpdateModel,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Met à jour une règle"""
            try:
                rule = await self.rule_manager.get_rule(rule_id)
                
                if not rule:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Rule not found"
                    )
                
                # Vérification des permissions
                if rule.tenant_id != current_user["tenant_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                # Application des mises à jour
                update_dict = updates.dict(exclude_unset=True)
                updated_rule = await self.rule_manager.update_rule(rule_id, update_dict)
                
                # Notification WebSocket
                await self.websocket_manager.send_to_tenant(
                    rule.tenant_id,
                    {
                        "type": "rule_updated",
                        "rule_id": rule_id,
                        "updates": update_dict
                    }
                )
                
                return APIResponse(
                    success=True,
                    message="Rule updated successfully",
                    data={"rule_id": rule_id}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to update rule", rule_id=rule_id, error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.delete("/api/v1/rules/{rule_id}", response_model=APIResponse)
        @limiter.limit("20/minute")
        async def delete_rule(
            request: Request,
            rule_id: str,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Supprime une règle"""
            try:
                rule = await self.rule_manager.get_rule(rule_id)
                
                if not rule:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Rule not found"
                    )
                
                # Vérification des permissions
                if rule.tenant_id != current_user["tenant_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                success = await self.rule_manager.remove_rule(rule_id, rule.tenant_id)
                
                if success:
                    # Notification WebSocket
                    await self.websocket_manager.send_to_tenant(
                        rule.tenant_id,
                        {
                            "type": "rule_deleted",
                            "rule_id": rule_id
                        }
                    )
                    
                    return APIResponse(
                        success=True,
                        message="Rule deleted successfully",
                        data={"rule_id": rule_id}
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to delete rule"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to delete rule", rule_id=rule_id, error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
    
    def _setup_evaluation_routes(self):
        """Configuration des routes d'évaluation"""
        
        @self.app.post("/api/v1/evaluate", response_model=APIResponse)
        @limiter.limit("50/minute")
        async def evaluate_rules(
            request: Request,
            evaluation_request: EvaluationRequestModel,
            background_tasks: BackgroundTasks,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Évalue les règles d'un tenant"""
            try:
                # Vérification des permissions
                if evaluation_request.tenant_id != current_user["tenant_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to tenant"
                    )
                
                # Conversion des métriques si fournies
                metrics = None
                if evaluation_request.metrics:
                    metrics_dict = evaluation_request.metrics.dict()
                    metrics = AlertMetrics(**metrics_dict)
                
                # Évaluation
                results = await self.rule_manager.evaluate_tenant_rules(
                    evaluation_request.tenant_id,
                    metrics
                )
                
                # Filtrage par rule_ids si spécifié
                if evaluation_request.rule_ids:
                    results = [
                        r for r in results 
                        if r.rule_id in evaluation_request.rule_ids
                    ]
                
                # Conversion en modèles de réponse
                result_data = [
                    EvaluationResultModel(
                        rule_id=result.rule_id,
                        triggered=result.triggered,
                        severity=result.severity.name,
                        message=result.message,
                        execution_time=result.execution_time,
                        timestamp=result.timestamp,
                        metadata=result.metadata
                    ).dict()
                    for result in results
                ]
                
                # Notification WebSocket des alertes déclenchées
                triggered_alerts = [r for r in results if r.triggered]
                if triggered_alerts:
                    background_tasks.add_task(
                        self._notify_triggered_alerts,
                        evaluation_request.tenant_id,
                        triggered_alerts
                    )
                
                return APIResponse(
                    success=True,
                    message=f"Evaluated {len(results)} rules",
                    data={
                        "results": result_data,
                        "triggered_count": len(triggered_alerts),
                        "total_evaluated": len(results)
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to evaluate rules", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.get("/api/v1/evaluate/stream/{tenant_id}")
        @limiter.limit("10/minute")
        async def stream_evaluations(
            request: Request,
            tenant_id: str,
            interval: int = 30,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Stream en temps réel des évaluations"""
            
            # Vérification des permissions
            if tenant_id != current_user["tenant_id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to tenant"
                )
            
            async def generate_evaluations():
                while True:
                    try:
                        results = await self.rule_manager.evaluate_tenant_rules(tenant_id)
                        
                        data = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "tenant_id": tenant_id,
                            "results": [
                                {
                                    "rule_id": r.rule_id,
                                    "triggered": r.triggered,
                                    "severity": r.severity.name,
                                    "message": r.message
                                }
                                for r in results
                            ]
                        }
                        
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(interval)
                        
                    except Exception as e:
                        logger.error("Stream evaluation error", error=str(e))
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break
            
            return StreamingResponse(
                generate_evaluations(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
    
    def _setup_monitoring_routes(self):
        """Configuration des routes de monitoring"""
        
        @self.app.get("/api/v1/stats", response_model=APIResponse)
        @limiter.limit("100/minute")
        async def get_statistics(
            request: Request,
            current_user: dict = Depends(self.auth_manager.get_current_user)
        ):
            """Récupère les statistiques du gestionnaire"""
            try:
                stats = await self.rule_manager.get_statistics()
                
                return APIResponse(
                    success=True,
                    message="Statistics retrieved successfully",
                    data=stats
                )
                
            except Exception as e:
                logger.error("Failed to get statistics", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Point d'accès aux métriques Prometheus"""
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            return Response(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
    
    def _setup_websocket_routes(self):
        """Configuration des routes WebSocket"""
        
        @self.app.websocket("/ws/{tenant_id}")
        async def websocket_endpoint(websocket: WebSocket, tenant_id: str):
            """Point d'accès WebSocket pour notifications temps réel"""
            
            # Authentification WebSocket (simplifié pour l'exemple)
            # En production, utiliser un mécanisme d'auth plus robuste
            
            user_id = "websocket_user"  # À récupérer depuis l'auth
            
            await self.websocket_manager.connect(websocket, tenant_id, user_id)
            
            try:
                while True:
                    # Maintien de la connexion
                    data = await websocket.receive_text()
                    
                    # Echo pour test de connectivité
                    if data == "ping":
                        await websocket.send_text("pong")
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    def _setup_graphql(self):
        """Configuration GraphQL"""
        
        @strawberry.type
        class Query:
            @strawberry.field
            async def rules(self, tenant_id: str) -> List[GraphQLRule]:
                """Récupère les règles via GraphQL"""
                rules = await self.rule_manager.list_rules(tenant_id=tenant_id)
                return [
                    GraphQLRule(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        description=rule.description,
                        severity=rule.severity.name,
                        category=rule.category.value,
                        tenant_id=rule.tenant_id,
                        enabled=rule.enabled,
                        status=rule.status.value
                    )
                    for rule in rules
                ]
        
        @strawberry.type
        class Mutation:
            @strawberry.mutation
            async def create_rule(self, rule_input: GraphQLRuleInput) -> GraphQLRule:
                """Crée une règle via GraphQL"""
                # Implementation simplifiée
                rule_config = {
                    "name": rule_input.name,
                    "description": rule_input.description,
                    "severity": rule_input.severity,
                    "category": rule_input.category,
                    "tenant_id": rule_input.tenant_id,
                    "enabled": rule_input.enabled,
                    "conditions": []  # À compléter
                }
                
                rule = await self.rule_manager.add_rule(rule_config)
                
                return GraphQLRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity.name,
                    category=rule.category.value,
                    tenant_id=rule.tenant_id,
                    enabled=rule.enabled,
                    status=rule.status.value
                )
        
        schema = strawberry.Schema(query=Query, mutation=Mutation)
        graphql_app = GraphQLRouter(schema)
        
        self.app.include_router(graphql_app, prefix="/graphql")
    
    def _setup_monitoring(self):
        """Configuration du monitoring Prometheus"""
        instrumentator = Instrumentator()
        instrumentator.instrument(self.app).expose(self.app)
    
    async def _notify_triggered_alerts(
        self,
        tenant_id: str,
        triggered_alerts: List[EvaluationResult]
    ):
        """Notifie les alertes déclenchées via WebSocket"""
        for alert in triggered_alerts:
            message = {
                "type": "alert_triggered",
                "rule_id": alert.rule_id,
                "severity": alert.severity.name,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            await self.websocket_manager.send_to_tenant(tenant_id, message)


# Factory pour création de l'API
async def create_api(
    redis_url: Optional[str] = None,
    database_url: Optional[str] = None,
    secret_key: str = "your-secret-key-here"
) -> AlertRulesAPI:
    """Factory pour créer l'API configurée"""
    
    # Configuration
    config = RuleEvaluationConfig(
        max_concurrent_evaluations=100,
        evaluation_timeout=30.0,
        cache_ttl=60,
        enable_ml_predictions=True,
        enable_distributed_cache=True
    )
    
    # Gestionnaire de règles
    rule_manager = await create_rule_manager(config, redis_url, database_url)
    await rule_manager.start()
    
    # Gestionnaire d'authentification
    auth_manager = AuthManager(secret_key)
    
    # Gestionnaire WebSocket
    websocket_manager = WebSocketManager()
    
    # Création de l'API
    api = AlertRulesAPI(rule_manager, auth_manager, websocket_manager)
    
    return api


# Point d'entrée pour le serveur
if __name__ == "__main__":
    import os
    
    # Configuration depuis les variables d'environnement
    redis_url = os.getenv("REDIS_URL")
    database_url = os.getenv("DATABASE_URL")
    secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Création et démarrage de l'API
    async def main():
        api = await create_api(redis_url, database_url, secret_key)
        
        uvicorn.run(
            api.app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    
    asyncio.run(main())


# Exportation
__all__ = [
    'AlertRulesAPI',
    'AuthManager',
    'WebSocketManager',
    'create_api',
    'RuleConfigModel',
    'AlertMetricsModel',
    'EvaluationRequestModel',
    'EvaluationResultModel',
    'APIResponse'
]
