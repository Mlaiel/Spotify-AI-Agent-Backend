"""
API FastAPI pour les algorithmes de monitoring et d'alertes.

Cette API expose tous les algorithmes via des endpoints REST :
- Détection d'anomalies en temps réel
- Classification d'alertes intelligente
- Analyse de corrélations avancée
- Modèles de prédiction d'incidents
- Analyse comportementale avancée

API haute performance avec authentification, rate limiting et monitoring.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, Depends, BackgroundTasks,
    status, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
import logging

from .config import AlgorithmType, Environment
from .factory import (
    AlgorithmManager, 
    create_default_dependencies,
    get_algorithm_manager
)
from .utils import METRICS_MANAGER

logger = logging.getLogger(__name__)

# Modèles Pydantic pour l'API
class AlgorithmRequest(BaseModel):
    """Requête générique pour un algorithme."""
    algorithm_type: str = Field(..., description="Type d'algorithme")
    model_name: str = Field(..., description="Nom du modèle")
    data: Dict[str, Any] = Field(..., description="Données à traiter")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Paramètres additionnels")
    
    @validator('algorithm_type')
    def validate_algorithm_type(cls, v):
        try:
            AlgorithmType(v)
        except ValueError:
            raise ValueError(f"Invalid algorithm type: {v}")
        return v

class BatchRequest(BaseModel):
    """Requête pour traitement par lots."""
    requests: List[AlgorithmRequest] = Field(..., description="Liste des requêtes")
    parallel_execution: bool = Field(default=True, description="Exécution parallèle")

class TrainingRequest(BaseModel):
    """Requête d'entraînement."""
    algorithm_type: str
    model_name: str
    training_data: Dict[str, Any]
    validation_data: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('algorithm_type')
    def validate_algorithm_type(cls, v):
        try:
            AlgorithmType(v)
        except ValueError:
            raise ValueError(f"Invalid algorithm type: {v}")
        return v

class AnomalyDetectionRequest(BaseModel):
    """Requête de détection d'anomalies."""
    metrics: List[Dict[str, Any]] = Field(..., description="Métriques à analyser")
    model_name: str = Field(default="ensemble", description="Modèle à utiliser")
    threshold: Optional[float] = Field(default=None, description="Seuil de détection")
    window_size: Optional[int] = Field(default=24, description="Taille de fenêtre")

class AlertClassificationRequest(BaseModel):
    """Requête de classification d'alertes."""
    alerts: List[Dict[str, Any]] = Field(..., description="Alertes à classifier")
    model_name: str = Field(default="ensemble_classifier", description="Modèle à utiliser")
    include_business_impact: bool = Field(default=True, description="Inclure l'impact business")

class CorrelationAnalysisRequest(BaseModel):
    """Requête d'analyse de corrélations."""
    metrics: List[Dict[str, Any]] = Field(..., description="Métriques à corréler")
    events: Optional[List[Dict[str, Any]]] = Field(default=None, description="Événements à analyser")
    time_window_hours: int = Field(default=24, description="Fenêtre temporelle en heures")
    correlation_threshold: float = Field(default=0.3, description="Seuil de corrélation")

class PredictionRequest(BaseModel):
    """Requête de prédiction."""
    historical_data: List[Dict[str, Any]] = Field(..., description="Données historiques")
    prediction_horizon_hours: int = Field(default=24, description="Horizon de prédiction")
    model_name: str = Field(default="incident_predictor", description="Modèle à utiliser")
    confidence_level: float = Field(default=0.95, description="Niveau de confiance")

class BehaviorAnalysisRequest(BaseModel):
    """Requête d'analyse comportementale."""
    user_events: Optional[List[Dict[str, Any]]] = Field(default=None, description="Événements utilisateur")
    system_metrics: Optional[List[Dict[str, Any]]] = Field(default=None, description="Métriques système")
    analysis_window_hours: int = Field(default=24, description="Fenêtre d'analyse")
    sensitivity: float = Field(default=0.8, description="Sensibilité de détection")

class AlgorithmResponse(BaseModel):
    """Réponse générique d'algorithme."""
    success: bool
    algorithm_type: str
    model_name: str
    result: Dict[str, Any]
    execution_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    error: str
    error_type: str
    algorithm_type: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: datetime

# Gestionnaire d'authentification simple
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie le token d'authentification."""
    # Implémentation basique - à remplacer par une vraie authentification
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Gestionnaire de cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application."""
    
    # Initialisation
    logger.info("Initializing algorithm manager...")
    dependencies = await create_default_dependencies()
    manager = get_algorithm_manager(dependencies)
    
    # Stockage dans l'état de l'application
    app.state.algorithm_manager = manager
    
    try:
        # Initialisation des algorithmes activés
        await manager.initialize_enabled_algorithms()
        logger.info("Algorithm manager initialized successfully")
        
        yield
        
    finally:
        # Nettoyage
        logger.info("Shutting down algorithm manager...")
        await manager.shutdown()
        logger.info("Algorithm manager shutdown complete")

# Création de l'application FastAPI
app = FastAPI(
    title="Spotify AI Agent - Algorithm API",
    description="API avancée pour les algorithmes de monitoring et d'alertes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware de logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log des requêtes HTTP."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Routes de santé et métriques
@app.get("/health", 
         response_model=Dict[str, Any],
         summary="Vérification de santé de l'API")
async def health_check():
    """Endpoint de vérification de santé."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "algorithms_active": len(await app.state.algorithm_manager.factory.list_active_algorithms())
    }

@app.get("/metrics", 
         response_class=PlainTextResponse,
         summary="Métriques Prometheus")
async def get_metrics():
    """Endpoint pour les métriques Prometheus."""
    return METRICS_MANAGER.export_metrics()

@app.get("/status",
         response_model=Dict[str, Any],
         summary="Statut détaillé des algorithmes")
async def get_status(token: str = Depends(verify_token)):
    """Endpoint pour le statut détaillé des algorithmes."""
    status = await app.state.algorithm_manager.get_algorithm_status()
    
    return {
        "algorithms": status,
        "total_active": len(status),
        "timestamp": datetime.now()
    }

# Routes pour la détection d'anomalies
@app.post("/anomaly-detection",
          response_model=AlgorithmResponse,
          summary="Détection d'anomalies")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Détecte les anomalies dans les métriques."""
    
    start_time = datetime.now()
    
    try:
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=AlgorithmType.ANOMALY_DETECTION,
            model_name=request.model_name,
            data={
                'metrics': request.metrics,
                'threshold': request.threshold,
                'window_size': request.window_size
            }
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=AlgorithmType.ANOMALY_DETECTION.value,
            model_name=request.model_name,
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )

# Routes pour la classification d'alertes
@app.post("/alert-classification",
          response_model=AlgorithmResponse,
          summary="Classification d'alertes")
async def classify_alerts(
    request: AlertClassificationRequest,
    token: str = Depends(verify_token)
):
    """Classifie les alertes selon leur priorité et impact."""
    
    start_time = datetime.now()
    
    try:
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=AlgorithmType.ALERT_CLASSIFICATION,
            model_name=request.model_name,
            data={
                'alerts': request.alerts,
                'include_business_impact': request.include_business_impact
            }
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=AlgorithmType.ALERT_CLASSIFICATION.value,
            model_name=request.model_name,
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Alert classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert classification failed: {str(e)}"
        )

# Routes pour l'analyse de corrélations
@app.post("/correlation-analysis",
          response_model=AlgorithmResponse,
          summary="Analyse de corrélations")
async def analyze_correlations(
    request: CorrelationAnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyse les corrélations entre métriques et événements."""
    
    start_time = datetime.now()
    
    try:
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=AlgorithmType.CORRELATION_ENGINE,
            model_name="correlation_engine",
            data={
                'metrics': request.metrics,
                'events': request.events,
                'time_window_hours': request.time_window_hours,
                'correlation_threshold': request.correlation_threshold
            }
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=AlgorithmType.CORRELATION_ENGINE.value,
            model_name="correlation_engine",
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Correlation analysis failed: {str(e)}"
        )

# Routes pour les prédictions
@app.post("/prediction",
          response_model=AlgorithmResponse,
          summary="Prédiction d'incidents")
async def predict_incidents(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Prédit les incidents futurs basés sur les données historiques."""
    
    start_time = datetime.now()
    
    try:
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=AlgorithmType.PREDICTION_MODELS,
            model_name=request.model_name,
            data={
                'historical_data': request.historical_data,
                'prediction_horizon_hours': request.prediction_horizon_hours,
                'confidence_level': request.confidence_level
            }
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=AlgorithmType.PREDICTION_MODELS.value,
            model_name=request.model_name,
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Routes pour l'analyse comportementale
@app.post("/behavior-analysis",
          response_model=AlgorithmResponse,
          summary="Analyse comportementale")
async def analyze_behavior(
    request: BehaviorAnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyse le comportement des utilisateurs et du système."""
    
    start_time = datetime.now()
    
    try:
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=AlgorithmType.BEHAVIORAL_ANALYSIS,
            model_name="behavior_analysis_engine",
            data={
                'user_events': request.user_events,
                'system_metrics': request.system_metrics,
                'analysis_window_hours': request.analysis_window_hours,
                'sensitivity': request.sensitivity
            }
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=AlgorithmType.BEHAVIORAL_ANALYSIS.value,
            model_name="behavior_analysis_engine",
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Behavior analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Behavior analysis failed: {str(e)}"
        )

# Route générique pour traitement d'algorithme
@app.post("/algorithm/process",
          response_model=AlgorithmResponse,
          summary="Traitement générique d'algorithme")
async def process_algorithm(
    request: AlgorithmRequest,
    token: str = Depends(verify_token)
):
    """Traite des données avec un algorithme spécifique."""
    
    start_time = datetime.now()
    
    try:
        algorithm_type = AlgorithmType(request.algorithm_type)
        
        result = await app.state.algorithm_manager.process_with_algorithm(
            algorithm_type=algorithm_type,
            model_name=request.model_name,
            data=request.data,
            **request.parameters
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AlgorithmResponse(
            success=True,
            algorithm_type=request.algorithm_type,
            model_name=request.model_name,
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Algorithm processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Algorithm processing failed: {str(e)}"
        )

# Route pour traitement par lots
@app.post("/batch/process",
          response_model=List[AlgorithmResponse],
          summary="Traitement par lots")
async def process_batch(
    request: BatchRequest,
    token: str = Depends(verify_token)
):
    """Traite plusieurs requêtes en parallèle."""
    
    if not request.parallel_execution:
        # Traitement séquentiel
        results = []
        for req in request.requests:
            try:
                result = await process_algorithm(req, token)
                results.append(result)
            except HTTPException as e:
                # Conversion de l'erreur HTTP en réponse d'erreur
                error_response = AlgorithmResponse(
                    success=False,
                    algorithm_type=req.algorithm_type,
                    model_name=req.model_name,
                    result={"error": e.detail},
                    execution_time_ms=0,
                    timestamp=datetime.now()
                )
                results.append(error_response)
        return results
    
    # Traitement parallèle
    batch_requests = []
    for req in request.requests:
        batch_requests.append({
            'algorithm_type': req.algorithm_type,
            'model_name': req.model_name,
            'data': req.data,
            'kwargs': req.parameters
        })
    
    try:
        results = await app.state.algorithm_manager.batch_process(batch_requests)
        
        # Conversion des résultats
        responses = []
        for i, result in enumerate(results):
            req = request.requests[i]
            
            if isinstance(result, Exception):
                response = AlgorithmResponse(
                    success=False,
                    algorithm_type=req.algorithm_type,
                    model_name=req.model_name,
                    result={"error": str(result)},
                    execution_time_ms=0,
                    timestamp=datetime.now()
                )
            else:
                response = AlgorithmResponse(
                    success=True,
                    algorithm_type=req.algorithm_type,
                    model_name=req.model_name,
                    result=result,
                    execution_time_ms=0,  # Non calculé pour le batch
                    timestamp=datetime.now()
                )
            
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )

# Route pour l'entraînement
@app.post("/train",
          response_model=Dict[str, Any],
          summary="Entraînement d'algorithme")
async def train_algorithm(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Entraîne un algorithme avec de nouvelles données."""
    
    try:
        algorithm_type = AlgorithmType(request.algorithm_type)
        
        # Entraînement en arrière-plan
        background_tasks.add_task(
            app.state.algorithm_manager.train_algorithm,
            algorithm_type,
            request.model_name,
            request.training_data,
            **request.parameters
        )
        
        return {
            "message": "Training started",
            "algorithm_type": request.algorithm_type,
            "model_name": request.model_name,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

# Route pour recharger les algorithmes
@app.post("/reload",
          response_model=Dict[str, Any],
          summary="Rechargement des algorithmes")
async def reload_algorithms(
    algorithm_type: Optional[str] = None,
    model_name: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Recharge les algorithmes (hot-reload)."""
    
    try:
        if algorithm_type and model_name:
            # Recharge un algorithme spécifique
            algo_type = AlgorithmType(algorithm_type)
            await app.state.algorithm_manager.factory.reload_algorithm(algo_type, model_name)
            message = f"Reloaded {algorithm_type}:{model_name}"
        else:
            # Recharge tous les algorithmes
            await app.state.algorithm_manager.factory.reload_all_algorithms()
            message = "Reloaded all algorithms"
        
        return {
            "message": message,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reload failed: {str(e)}"
        )

# Gestionnaire d'erreurs global
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire d'erreurs HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException",
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs général."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_type=type(exc).__name__,
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    import os
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Lancement du serveur
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,  # Pas de reload en production
        access_log=True
    )
