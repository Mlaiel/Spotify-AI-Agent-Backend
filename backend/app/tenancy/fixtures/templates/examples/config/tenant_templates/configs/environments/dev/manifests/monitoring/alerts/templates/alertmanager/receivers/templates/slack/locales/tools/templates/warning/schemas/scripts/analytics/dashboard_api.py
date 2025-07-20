#!/usr/bin/env python3
"""
Analytics Dashboard API - API du Tableau de Bord Analytics
========================================================

API REST avancée pour le tableau de bord analytics du Spotify AI Agent.
Fournit des endpoints pour visualiser, configurer et interagir avec
le système analytics en temps réel.

Fonctionnalités:
- API REST complète pour metrics, dashboards, alertes
- WebSocket en temps réel pour streaming de données
- Authentification et autorisation
- Pagination et filtrage avancés
- Cache intelligent
- Rate limiting
- Documentation API automatique

Endpoints principaux:
- /api/v1/metrics/* - Gestion des métriques
- /api/v1/dashboards/* - Tableaux de bord
- /api/v1/alerts/* - Système d'alertes
- /api/v1/ml/* - Modèles ML et prédictions
- /api/v1/performance/* - Monitoring performances
- /ws/* - WebSocket temps réel

Usage:
    uvicorn dashboard_api:app --host 0.0.0.0 --port 8000

Auteur: Fahed Mlaiel - Lead Full-Stack Developer & API Architect
Équipe: Backend Engineers, Frontend Developers, UX/UI Designers
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import jwt
from passlib.context import CryptContext
import aioredis
from contextlib import asynccontextmanager

# Analytics modules
from config import AnalyticsConfig, get_config
from core import AnalyticsEngine, MetricsCollector, AlertManager
from models import Metric, Event, Alert, Dashboard, create_metric, create_event
from ml import ModelManager, MLPrediction
from storage import StorageManager
from utils import Logger, RateLimiter, Timer, Formatter
from performance_monitor import PerformanceMonitor


# Configuration FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application."""
    # Startup
    logger = Logger("DashboardAPI")
    logger.info("Démarrage de l'API Dashboard...")
    
    # Initialiser les services
    await app.state.analytics_engine.start()
    await app.state.performance_monitor.start_monitoring()
    
    logger.info("API Dashboard démarrée avec succès")
    
    yield
    
    # Shutdown
    logger.info("Arrêt de l'API Dashboard...")
    await app.state.analytics_engine.stop()
    await app.state.performance_monitor.stop_monitoring()
    logger.info("API Dashboard arrêtée")


# Application FastAPI
app = FastAPI(
    title="Spotify AI Analytics Dashboard API",
    description="API avancée pour le système analytics Spotify AI Agent",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configuration
config = get_config()
logger = Logger("DashboardAPI")

# Services globaux
analytics_engine = AnalyticsEngine(config)
performance_monitor = PerformanceMonitor(config)
storage_manager = StorageManager(config)
model_manager = ModelManager(config)

# État de l'application
app.state.analytics_engine = analytics_engine
app.state.performance_monitor = performance_monitor
app.state.storage_manager = storage_manager
app.state.model_manager = model_manager

# Authentification
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cache et Rate Limiting
rate_limiter = RateLimiter()
redis_client = None

# WebSocket Manager
class WebSocketManager:
    """Gestionnaire de connexions WebSocket."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = defaultdict(list)
        self.logger = Logger("WebSocketManager")
    
    async def connect(self, websocket: WebSocket, topic: str = "general"):
        """Connecte un client WebSocket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[topic].append(websocket)
        self.logger.info(f"Client connecté à {topic}. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Déconnecte un client WebSocket."""
        self.active_connections.remove(websocket)
        for topic, connections in self.subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
        self.logger.info(f"Client déconnecté. Total: {len(self.active_connections)}")
    
    async def send_to_topic(self, topic: str, message: dict):
        """Envoie un message à tous les clients d'un topic."""
        if topic in self.subscriptions:
            disconnected = []
            for websocket in self.subscriptions[topic]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
            
            # Nettoyer les connexions fermées
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast(self, message: dict):
        """Diffuse un message à tous les clients connectés."""
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Nettoyer les connexions fermées
        for ws in disconnected:
            self.disconnect(ws)

websocket_manager = WebSocketManager()


# Modèles de données API
class MetricRequest(BaseModel):
    """Requête de création de métrique."""
    name: str = Field(..., description="Nom de la métrique")
    value: float = Field(..., description="Valeur de la métrique")
    tenant_id: str = Field(..., description="ID du tenant")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags de la métrique")
    timestamp: Optional[datetime] = Field(None, description="Timestamp (auto si non fourni)")


class DashboardRequest(BaseModel):
    """Requête de création de tableau de bord."""
    name: str = Field(..., description="Nom du dashboard")
    description: str = Field("", description="Description du dashboard")
    tenant_id: str = Field(..., description="ID du tenant")
    layout: Dict[str, Any] = Field(default_factory=dict, description="Configuration layout")
    widgets: List[Dict[str, Any]] = Field(default_factory=list, description="Liste des widgets")
    is_public: bool = Field(False, description="Dashboard public")


class AlertRequest(BaseModel):
    """Requête de création d'alerte."""
    name: str = Field(..., description="Nom de l'alerte")
    description: str = Field("", description="Description de l'alerte")
    tenant_id: str = Field(..., description="ID du tenant")
    condition: Dict[str, Any] = Field(..., description="Condition de déclenchement")
    severity: str = Field("warning", description="Niveau de sévérité")
    notification_channels: List[str] = Field(default_factory=list, description="Canaux de notification")


class PredictionRequest(BaseModel):
    """Requête de prédiction ML."""
    model_name: str = Field(..., description="Nom du modèle")
    features: Dict[str, Any] = Field(..., description="Features pour la prédiction")
    tenant_id: str = Field(..., description="ID du tenant")


class QueryFilter(BaseModel):
    """Filtre de requête avancé."""
    tenant_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None
    limit: int = Field(100, ge=1, le=10000)
    offset: int = Field(0, ge=0)
    sort_by: str = Field("timestamp")
    sort_order: str = Field("desc", regex="^(asc|desc)$")


# Dépendances
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obtient l'utilisateur actuel à partir du token JWT."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, config.security.secret_key, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token invalide")
        return {"user_id": user_id, "tenant_id": payload.get("tenant_id")}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide")


async def check_rate_limit(user: dict = Depends(get_current_user)):
    """Vérification du rate limiting."""
    user_id = user["user_id"]
    if not await rate_limiter.check_rate_limit(f"api:{user_id}", limit=100, window=60):
        raise HTTPException(status_code=429, detail="Rate limit dépassé")
    return user


# Endpoints d'authentification
@app.post("/api/v1/auth/login")
async def login(username: str = Body(...), password: str = Body(...)):
    """Authentification utilisateur."""
    # Simulation d'authentification
    if username == "admin" and password == "admin123":
        token_data = {
            "sub": "admin",
            "tenant_id": "default",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(token_data, config.security.secret_key, algorithm="HS256")
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Identifiants invalides")


@app.post("/api/v1/auth/refresh")
async def refresh_token(user: dict = Depends(get_current_user)):
    """Renouvellement du token."""
    token_data = {
        "sub": user["user_id"],
        "tenant_id": user["tenant_id"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    token = jwt.encode(token_data, config.security.secret_key, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}


# Endpoints métriques
@app.post("/api/v1/metrics")
async def create_metric(
    metric_data: MetricRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(check_rate_limit)
):
    """Créer une nouvelle métrique."""
    try:
        # Créer la métrique
        metric = create_metric(
            name=metric_data.name,
            value=metric_data.value,
            tenant_id=metric_data.tenant_id or user["tenant_id"],
            tags=metric_data.tags,
            timestamp=metric_data.timestamp
        )
        
        # Traitement en arrière-plan
        background_tasks.add_task(
            analytics_engine.metrics_collector.collect_metric,
            metric.tenant_id,
            metric.name,
            metric.value,
            metric.tags
        )
        
        # Notification WebSocket
        await websocket_manager.send_to_topic("metrics", {
            "type": "metric_created",
            "data": metric.dict()
        })
        
        return {"status": "success", "metric_id": metric.id}
        
    except Exception as e:
        logger.error(f"Erreur création métrique: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
async def get_metrics(
    tenant_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user: dict = Depends(check_rate_limit)
):
    """Récupérer les métriques avec filtrage."""
    try:
        # Filtres
        filters = {
            "tenant_id": tenant_id or user["tenant_id"],
            "start_time": start_time,
            "end_time": end_time,
            "name": name,
            "limit": limit,
            "offset": offset
        }
        
        # Simulation de données (à remplacer par vraie requête DB)
        metrics = []
        for i in range(min(limit, 20)):
            metric = create_metric(
                name=name or f"sample_metric_{i}",
                value=float(i * 10),
                tenant_id=filters["tenant_id"],
                tags={"source": f"server_{i % 3}"}
            )
            metrics.append(metric.dict())
        
        return {
            "data": metrics,
            "total": len(metrics),
            "limit": limit,
            "offset": offset,
            "filters": filters
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération métriques: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/{metric_id}")
async def get_metric(
    metric_id: str = Path(...),
    user: dict = Depends(check_rate_limit)
):
    """Récupérer une métrique spécifique."""
    try:
        # Simulation
        metric = create_metric(
            name="sample_metric",
            value=42.0,
            tenant_id=user["tenant_id"],
            tags={"id": metric_id}
        )
        
        return {"data": metric.dict()}
        
    except Exception as e:
        logger.error(f"Erreur récupération métrique {metric_id}: {e}")
        raise HTTPException(status_code=404, detail="Métrique non trouvée")


@app.get("/api/v1/metrics/aggregated")
async def get_aggregated_metrics(
    metric_name: str = Query(...),
    aggregation: str = Query("avg", regex="^(avg|sum|min|max|count)$"),
    granularity: str = Query("1h", regex="^(1m|5m|15m|1h|1d)$"),
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    user: dict = Depends(check_rate_limit)
):
    """Récupérer des métriques agrégées."""
    try:
        # Simulation de données agrégées
        time_points = []
        current_time = start_time
        
        while current_time <= end_time:
            time_points.append({
                "timestamp": current_time.isoformat(),
                "value": float(hash(str(current_time)) % 100),
                "aggregation": aggregation
            })
            
            # Incrément basé sur la granularité
            if granularity == "1m":
                current_time += timedelta(minutes=1)
            elif granularity == "5m":
                current_time += timedelta(minutes=5)
            elif granularity == "15m":
                current_time += timedelta(minutes=15)
            elif granularity == "1h":
                current_time += timedelta(hours=1)
            else:  # 1d
                current_time += timedelta(days=1)
        
        return {
            "metric_name": metric_name,
            "aggregation": aggregation,
            "granularity": granularity,
            "data": time_points[:100]  # Limiter pour éviter surcharge
        }
        
    except Exception as e:
        logger.error(f"Erreur agrégation métriques: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints Dashboards
@app.post("/api/v1/dashboards")
async def create_dashboard(
    dashboard_data: DashboardRequest,
    user: dict = Depends(check_rate_limit)
):
    """Créer un nouveau tableau de bord."""
    try:
        # Créer le dashboard
        dashboard = Dashboard(
            name=dashboard_data.name,
            description=dashboard_data.description,
            tenant_id=dashboard_data.tenant_id or user["tenant_id"],
            layout=dashboard_data.layout,
            widgets=dashboard_data.widgets,
            is_public=dashboard_data.is_public,
            created_by=user["user_id"]
        )
        
        # Notification WebSocket
        await websocket_manager.send_to_topic("dashboards", {
            "type": "dashboard_created",
            "data": dashboard.dict()
        })
        
        return {"status": "success", "dashboard_id": dashboard.id}
        
    except Exception as e:
        logger.error(f"Erreur création dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboards")
async def get_dashboards(
    tenant_id: Optional[str] = Query(None),
    include_public: bool = Query(True),
    user: dict = Depends(check_rate_limit)
):
    """Récupérer les tableaux de bord."""
    try:
        # Simulation
        dashboards = []
        for i in range(5):
            dashboard = Dashboard(
                name=f"Dashboard {i+1}",
                description=f"Description du dashboard {i+1}",
                tenant_id=tenant_id or user["tenant_id"],
                layout={"columns": 2, "rows": 3},
                widgets=[
                    {"type": "chart", "title": f"Widget {j+1}", "size": "medium"}
                    for j in range(3)
                ],
                is_public=i % 2 == 0,
                created_by=user["user_id"]
            )
            dashboards.append(dashboard.dict())
        
        return {"data": dashboards}
        
    except Exception as e:
        logger.error(f"Erreur récupération dashboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints Alertes
@app.post("/api/v1/alerts")
async def create_alert(
    alert_data: AlertRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(check_rate_limit)
):
    """Créer une nouvelle alerte."""
    try:
        # Créer l'alerte
        alert = Alert(
            name=alert_data.name,
            description=alert_data.description,
            tenant_id=alert_data.tenant_id or user["tenant_id"],
            condition=alert_data.condition,
            severity=alert_data.severity,
            notification_channels=alert_data.notification_channels,
            is_active=True,
            created_by=user["user_id"]
        )
        
        # Enregistrer dans le gestionnaire d'alertes
        background_tasks.add_task(
            analytics_engine.alert_manager.add_alert_rule,
            alert
        )
        
        return {"status": "success", "alert_id": alert.id}
        
    except Exception as e:
        logger.error(f"Erreur création alerte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts")
async def get_alerts(
    status: Optional[str] = Query(None, regex="^(active|resolved|all)$"),
    severity: Optional[str] = Query(None, regex="^(info|warning|critical)$"),
    user: dict = Depends(check_rate_limit)
):
    """Récupérer les alertes."""
    try:
        # Récupérer depuis le gestionnaire d'alertes
        active_alerts = analytics_engine.alert_manager.active_alerts
        
        alerts = []
        for alert_id, alert in active_alerts.items():
            if severity and alert.severity != severity:
                continue
            
            alerts.append({
                "id": alert_id,
                "name": alert.name,
                "severity": alert.severity,
                "status": "active",
                "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None,
                "condition": alert.condition,
                "tenant_id": alert.tenant_id
            })
        
        return {"data": alerts}
        
    except Exception as e:
        logger.error(f"Erreur récupération alertes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints ML
@app.post("/api/v1/ml/predict")
async def predict(
    prediction_request: PredictionRequest,
    user: dict = Depends(check_rate_limit)
):
    """Faire une prédiction avec un modèle ML."""
    try:
        # Récupérer le modèle
        model = model_manager.get_model(prediction_request.model_name)
        if not model:
            raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
        # Faire la prédiction
        result = await model.predict([prediction_request.features])
        
        # Notification WebSocket
        await websocket_manager.send_to_topic("ml", {
            "type": "prediction_made",
            "data": {
                "model": prediction_request.model_name,
                "prediction": result.dict(),
                "tenant_id": prediction_request.tenant_id
            }
        })
        
        return {"status": "success", "prediction": result.dict()}
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/models")
async def get_models(user: dict = Depends(check_rate_limit)):
    """Récupérer la liste des modèles ML."""
    try:
        models_stats = model_manager.get_all_model_stats()
        
        models = []
        for model_name, stats in models_stats.items():
            models.append({
                "name": model_name,
                "is_trained": stats["is_trained"],
                "feature_count": stats["feature_count"],
                "accuracy": stats["metrics"]["accuracy"],
                "last_trained": stats["last_trained"],
                "model_size": stats["model_size"]
            })
        
        return {"data": models}
        
    except Exception as e:
        logger.error(f"Erreur récupération modèles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/models/{model_name}/train")
async def train_model(
    model_name: str = Path(...),
    background_tasks: BackgroundTasks,
    user: dict = Depends(check_rate_limit)
):
    """Entraîner un modèle ML."""
    try:
        model = model_manager.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
        # Entraînement en arrière-plan
        background_tasks.add_task(
            _train_model_background,
            model,
            model_name
        )
        
        return {"status": "training_started", "model": model_name}
        
    except Exception as e:
        logger.error(f"Erreur entraînement modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_model_background(model, model_name: str):
    """Entraînement de modèle en arrière-plan."""
    try:
        # Simulation de données d'entraînement
        training_data = [{"feature1": i, "feature2": i*2} for i in range(100)]
        
        await model.train(training_data)
        
        # Notification de fin d'entraînement
        await websocket_manager.send_to_topic("ml", {
            "type": "training_completed",
            "data": {"model": model_name, "status": "success"}
        })
        
    except Exception as e:
        logger.error(f"Erreur entraînement background: {e}")
        await websocket_manager.send_to_topic("ml", {
            "type": "training_failed",
            "data": {"model": model_name, "error": str(e)}
        })


# Endpoints Performance
@app.get("/api/v1/performance/status")
async def get_performance_status(user: dict = Depends(check_rate_limit)):
    """Récupérer le statut de performances."""
    try:
        # Collecter les métriques actuelles
        system_metrics = performance_monitor.collect_system_metrics()
        db_metrics = await performance_monitor.collect_database_metrics()
        ml_metrics = await performance_monitor.collect_ml_metrics()
        
        # Analyser les performances
        alerts = performance_monitor.analyze_performance()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent
            },
            "databases": [
                {
                    "name": db.database_name,
                    "response_time": db.response_time_avg,
                    "throughput": db.throughput_ops_per_sec
                }
                for db in db_metrics
            ],
            "ml_models": [
                {
                    "name": ml.model_name,
                    "latency": ml.prediction_latency_ms,
                    "accuracy": ml.accuracy_score
                }
                for ml in ml_metrics
            ],
            "alerts": len([a for a in alerts if a.alert_type in ["warning", "critical"]]),
            "status": "healthy" if system_metrics.cpu_percent < 70 else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Erreur statut performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/report")
async def get_performance_report(
    format: str = Query("json", regex="^(json|summary)$"),
    user: dict = Depends(check_rate_limit)
):
    """Générer un rapport de performances."""
    try:
        report = performance_monitor.generate_performance_report()
        
        if format == "summary":
            return {
                "summary": report["summary"],
                "recommendations": report["recommendations"][:5],  # Top 5
                "alerts_summary": report["alerts_summary"]
            }
        
        return report
        
    except Exception as e:
        logger.error(f"Erreur rapport performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoints
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket pour streaming des métriques en temps réel."""
    await websocket_manager.connect(websocket, "metrics")
    try:
        while True:
            # Attendre un message (keep-alive)
            await websocket.receive_text()
            
            # Envoyer des métriques en temps réel (simulation)
            metric = create_metric(
                name="realtime_metric",
                value=float(time.time() % 100),
                tenant_id="realtime",
                tags={"source": "websocket"}
            )
            
            await websocket.send_json({
                "type": "metric_update",
                "data": metric.dict()
            })
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/performance")
async def websocket_performance(websocket: WebSocket):
    """WebSocket pour monitoring des performances en temps réel."""
    await websocket_manager.connect(websocket, "performance")
    try:
        while True:
            await asyncio.sleep(5)  # Mise à jour toutes les 5 secondes
            
            # Collecter les métriques
            system_metrics = performance_monitor.collect_system_metrics()
            
            await websocket.send_json({
                "type": "performance_update",
                "data": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu": system_metrics.cpu_percent,
                    "memory": system_metrics.memory_percent,
                    "disk": system_metrics.disk_usage_percent
                }
            })
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# Endpoints de santé et monitoring
@app.get("/health")
async def health_check():
    """Vérification de santé de l'API."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "analytics_engine": analytics_engine.is_running if hasattr(analytics_engine, 'is_running') else True,
            "storage_manager": True,
            "model_manager": True,
            "performance_monitor": performance_monitor.is_monitoring
        }
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Endpoint Prometheus pour métriques."""
    # Simulation de métriques Prometheus
    metrics = f"""
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{{method="GET",endpoint="/api/v1/metrics"}} {hash("requests") % 1000}

# HELP api_response_time_seconds API response time
# TYPE api_response_time_seconds histogram
api_response_time_seconds_bucket{{le="0.1"}} {hash("bucket_01") % 100}
api_response_time_seconds_bucket{{le="0.5"}} {hash("bucket_05") % 200}
api_response_time_seconds_bucket{{le="1.0"}} {hash("bucket_10") % 300}

# HELP system_cpu_usage CPU usage percentage
# TYPE system_cpu_usage gauge
system_cpu_usage {performance_monitor.system_metrics_history[-1].cpu_percent if performance_monitor.system_metrics_history else 0}

# HELP system_memory_usage Memory usage percentage
# TYPE system_memory_usage gauge
system_memory_usage {performance_monitor.system_metrics_history[-1].memory_percent if performance_monitor.system_metrics_history else 0}
"""
    
    return StreamingResponse(
        iter([metrics]),
        media_type="text/plain"
    )


# Gestionnaire d'erreurs
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Erreur interne: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Une erreur interne s'est produite"}
    )


if __name__ == "__main__":
    import uvicorn
    
    print("""
    🚀 SPOTIFY AI ANALYTICS DASHBOARD API
    ====================================
    📊 API REST complète
    🔄 WebSocket temps réel
    🔐 Authentification JWT
    📈 Monitoring intégré
    🎯 Rate limiting
    📚 Documentation auto
    
    By Fahed Mlaiel & API Team
    """)
    
    uvicorn.run(
        "dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
