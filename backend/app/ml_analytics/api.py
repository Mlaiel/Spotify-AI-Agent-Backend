# 🎵 ML Analytics API Endpoints
# ============================
# 
# API REST avancée pour ML Analytics
# Endpoints enterprise avec authentification et validation
#
# 🎖️ Expert: Lead Dev + Développeur Backend Senior

"""
🌐 ML Analytics API Endpoints
=============================

Comprehensive REST API for ML Analytics:
- Model management and inference endpoints
- Analytics and reporting endpoints
- Real-time monitoring and health checks
- Data processing and pipeline management
- Security and authentication integration
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import io
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from ..core.dependencies import get_current_user, get_database
from ..schemas.base import BaseResponse, PaginationParams
from .core import MLAnalyticsEngine
from .models import SpotifyRecommendationModel
from .audio import AudioAnalysisModel
from .config import MLAnalyticsConfig
from .monitoring import ml_monitor
from .exceptions import MLAnalyticsError, ModelNotFoundError
from .utils import SecurityValidator, performance_monitor

# Configuration de l'API
router = APIRouter(prefix="/ml-analytics", tags=["ML Analytics"])
security = HTTPBearer()
config = MLAnalyticsConfig()

# Modèles Pydantic pour les requêtes/réponses
class RecommendationRequest(BaseModel):
    """Requête de recommandation"""
    user_id: str = Field(..., description="ID de l'utilisateur")
    track_ids: Optional[List[str]] = Field(None, description="IDs des tracks de référence")
    limit: int = Field(10, ge=1, le=100, description="Nombre de recommandations")
    include_features: bool = Field(False, description="Inclure les features audio")
    algorithm: str = Field("hybrid", description="Algorithme de recommandation")
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        allowed = ['content_based', 'collaborative', 'hybrid', 'deep_learning']
        if v not in allowed:
            raise ValueError(f"Algorithme doit être parmi: {', '.join(allowed)}")
        return v


class AudioAnalysisRequest(BaseModel):
    """Requête d'analyse audio"""
    track_id: Optional[str] = Field(None, description="ID du track")
    audio_url: Optional[str] = Field(None, description="URL de l'audio")
    file_path: Optional[str] = Field(None, description="Chemin du fichier audio")
    analysis_type: str = Field("complete", description="Type d'analyse")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed = ['basic', 'complete', 'genre', 'mood', 'quality']
        if v not in allowed:
            raise ValueError(f"Type d'analyse doit être parmi: {', '.join(allowed)}")
        return v


class ModelTrainingRequest(BaseModel):
    """Requête d'entraînement de modèle"""
    model_type: str = Field(..., description="Type de modèle")
    training_data_path: str = Field(..., description="Chemin des données d'entraînement")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Paramètres du modèle")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Proportion de validation")
    epochs: int = Field(10, ge=1, le=1000, description="Nombre d'époques")


class PipelineRequest(BaseModel):
    """Requête de pipeline"""
    pipeline_type: str = Field(..., description="Type de pipeline")
    data_source: str = Field(..., description="Source de données")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres du pipeline")
    schedule: Optional[str] = Field(None, description="Planification (cron format)")


class AnalyticsQuery(BaseModel):
    """Requête d'analytics"""
    metric_name: str = Field(..., description="Nom de la métrique")
    start_date: datetime = Field(..., description="Date de début")
    end_date: datetime = Field(..., description="Date de fin")
    granularity: str = Field("day", description="Granularité temporelle")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres")
    
    @validator('granularity')
    def validate_granularity(cls, v):
        allowed = ['hour', 'day', 'week', 'month']
        if v not in allowed:
            raise ValueError(f"Granularité doit être parmi: {', '.join(allowed)}")
        return v


# Réponses
class RecommendationResponse(BaseResponse):
    """Réponse de recommandation"""
    recommendations: List[Dict[str, Any]]
    algorithm_used: str
    execution_time_ms: float
    model_version: str


class AudioAnalysisResponse(BaseResponse):
    """Réponse d'analyse audio"""
    track_id: Optional[str]
    features: Dict[str, Any]
    genre_prediction: Optional[Dict[str, float]]
    mood_analysis: Optional[Dict[str, float]]
    quality_score: Optional[float]
    analysis_time_ms: float


class ModelStatusResponse(BaseResponse):
    """Réponse de statut de modèle"""
    model_id: str
    status: str
    version: str
    accuracy_metrics: Dict[str, float]
    last_training: Optional[datetime]
    next_scheduled_training: Optional[datetime]


class MonitoringResponse(BaseResponse):
    """Réponse de monitoring"""
    system_health: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]


# Dépendances
async def get_ml_engine() -> MLAnalyticsEngine:
    """Récupération de l'engine ML"""
    engine = MLAnalyticsEngine()
    await engine.initialize()
    return engine


async def validate_request_size(request_data: Any):
    """Validation de la taille de la requête"""
    if not SecurityValidator.validate_input_size(request_data, max_size_mb=10):
        raise HTTPException(
            status_code=413,
            detail="Taille de la requête trop importante (max 10MB)"
        )


# ===========================
# 🎵 Endpoints de Recommandation
# ===========================

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Génération de recommandations musicales personnalisées
    """
    async with ml_monitor.monitor_operation("get_recommendations"):
        try:
            # Validation de sécurité
            await validate_request_size(request.dict())
            
            # Récupération du modèle de recommandation
            recommendation_model = await engine.get_model("spotify_recommendation")
            
            # Génération des recommandations
            start_time = datetime.utcnow()
            
            recommendations = await recommendation_model.generate_recommendations(
                user_id=request.user_id,
                reference_tracks=request.track_ids,
                num_recommendations=request.limit,
                algorithm=request.algorithm,
                include_features=request.include_features
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Logging de l'utilisation
            background_tasks.add_task(
                engine.log_model_usage,
                "spotify_recommendation",
                current_user.get("id"),
                {"recommendations_count": len(recommendations)}
            )
            
            return RecommendationResponse(
                success=True,
                data={
                    "recommendations": recommendations,
                    "algorithm_used": request.algorithm,
                    "execution_time_ms": execution_time,
                    "model_version": recommendation_model.version
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{user_id}/history")
async def get_recommendation_history(
    user_id: str = Path(..., description="ID de l'utilisateur"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Historique des recommandations pour un utilisateur
    """
    try:
        history = await engine.get_recommendation_history(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "history": history,
                "total_count": len(history),
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": len(history) == limit
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 🎧 Endpoints d'Analyse Audio
# ===========================

@router.post("/audio/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    request: AudioAnalysisRequest,
    background_tasks: BackgroundTasks,
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyse avancée des caractéristiques audio
    """
    async with ml_monitor.monitor_operation("analyze_audio"):
        try:
            # Validation de sécurité
            await validate_request_size(request.dict())
            
            # Récupération du modèle d'analyse audio
            audio_model = await engine.get_model("audio_analysis")
            
            # Détermination de la source audio
            audio_source = None
            if request.track_id:
                audio_source = await engine.get_audio_source_by_track_id(request.track_id)
            elif request.audio_url:
                audio_source = request.audio_url
            elif request.file_path:
                if SecurityValidator.validate_model_path(request.file_path):
                    audio_source = request.file_path
                else:
                    raise HTTPException(status_code=400, detail="Chemin de fichier invalide")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Au moins une source audio doit être spécifiée"
                )
            
            # Analyse audio
            start_time = datetime.utcnow()
            
            analysis_result = await audio_model.analyze_audio(
                audio_source=audio_source,
                analysis_type=request.analysis_type
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Logging de l'utilisation
            background_tasks.add_task(
                engine.log_model_usage,
                "audio_analysis",
                current_user.get("id"),
                {"analysis_type": request.analysis_type}
            )
            
            return AudioAnalysisResponse(
                success=True,
                data={
                    "track_id": request.track_id,
                    "features": analysis_result.get("features", {}),
                    "genre_prediction": analysis_result.get("genre_prediction"),
                    "mood_analysis": analysis_result.get("mood_analysis"),
                    "quality_score": analysis_result.get("quality_score"),
                    "analysis_time_ms": execution_time
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/features/{track_id}")
async def get_cached_audio_features(
    track_id: str = Path(..., description="ID du track"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Récupération des features audio en cache
    """
    try:
        features = await engine.get_cached_audio_features(track_id)
        
        if not features:
            raise HTTPException(
                status_code=404,
                detail="Features audio non trouvées en cache"
            )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "track_id": track_id,
                "features": features,
                "cached_at": features.get("cached_at")
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 🤖 Endpoints de Gestion des Modèles
# ===========================

@router.get("/models", response_model=List[ModelStatusResponse])
async def list_models(
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Liste de tous les modèles ML disponibles
    """
    try:
        models = await engine.get_all_models()
        
        model_statuses = []
        for model_id, model_info in models.items():
            status = ModelStatusResponse(
                success=True,
                data={
                    "model_id": model_id,
                    "status": model_info.get("status", "unknown"),
                    "version": model_info.get("version", "unknown"),
                    "accuracy_metrics": model_info.get("metrics", {}),
                    "last_training": model_info.get("last_training"),
                    "next_scheduled_training": model_info.get("next_training")
                }
            )
            model_statuses.append(status)
        
        return model_statuses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_id: str = Path(..., description="ID du modèle"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Statut détaillé d'un modèle spécifique
    """
    try:
        model_info = await engine.get_model_info(model_id)
        
        if not model_info:
            raise ModelNotFoundError(model_id)
        
        return ModelStatusResponse(
            success=True,
            data={
                "model_id": model_id,
                "status": model_info.get("status", "unknown"),
                "version": model_info.get("version", "unknown"),
                "accuracy_metrics": model_info.get("metrics", {}),
                "last_training": model_info.get("last_training"),
                "next_scheduled_training": model_info.get("next_training")
            }
        )
        
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' non trouvé")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/train")
async def train_model(
    model_id: str = Path(..., description="ID du modèle"),
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Déclenchement de l'entraînement d'un modèle
    """
    try:
        # Validation de sécurité
        if not SecurityValidator.validate_model_path(request.training_data_path):
            raise HTTPException(
                status_code=400,
                detail="Chemin de données d'entraînement invalide"
            )
        
        # Déclenchement de l'entraînement en arrière-plan
        training_job_id = await engine.start_model_training(
            model_id=model_id,
            training_config={
                "data_path": request.training_data_path,
                "model_params": request.model_params,
                "validation_split": request.validation_split,
                "epochs": request.epochs,
                "user_id": current_user.get("id")
            }
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "training_job_id": training_job_id,
                "model_id": model_id,
                "status": "training_started",
                "estimated_duration_minutes": request.epochs * 2  # Estimation
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/training/{job_id}/status")
async def get_training_status(
    model_id: str = Path(..., description="ID du modèle"),
    job_id: str = Path(..., description="ID du job d'entraînement"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Statut d'un job d'entraînement
    """
    try:
        status = await engine.get_training_status(model_id, job_id)
        
        return JSONResponse(content={
            "success": True,
            "data": status
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 🔄 Endpoints de Pipeline
# ===========================

@router.post("/pipelines")
async def create_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Création et exécution d'un pipeline de données
    """
    try:
        pipeline_id = await engine.create_pipeline(
            pipeline_type=request.pipeline_type,
            data_source=request.data_source,
            parameters=request.parameters,
            schedule=request.schedule,
            created_by=current_user.get("id")
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "pipeline_id": pipeline_id,
                "status": "created",
                "type": request.pipeline_type
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines")
async def list_pipelines(
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Liste des pipelines
    """
    try:
        pipelines = await engine.get_pipelines(
            status_filter=status,
            limit=limit,
            offset=offset
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "pipelines": pipelines,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": len(pipelines) == limit
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}/status")
async def get_pipeline_status(
    pipeline_id: str = Path(..., description="ID du pipeline"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Statut d'un pipeline
    """
    try:
        status = await engine.get_pipeline_status(pipeline_id)
        
        return JSONResponse(content={
            "success": True,
            "data": status
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 📊 Endpoints d'Analytics
# ===========================

@router.post("/analytics/query")
async def query_analytics(
    request: AnalyticsQuery,
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Requête d'analytics avec agrégations temporelles
    """
    try:
        # Validation des dates
        if request.end_date <= request.start_date:
            raise HTTPException(
                status_code=400,
                detail="La date de fin doit être postérieure à la date de début"
            )
        
        # Exécution de la requête
        results = await engine.query_analytics(
            metric_name=request.metric_name,
            start_date=request.start_date,
            end_date=request.end_date,
            granularity=request.granularity,
            filters=request.filters
        )
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "metric_name": request.metric_name,
                "granularity": request.granularity,
                "results": results,
                "query_metadata": {
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "filters_applied": request.filters
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/export/{format}")
async def export_analytics(
    format: str = Path(..., description="Format d'export (csv, json, excel)"),
    metric_name: str = Query(..., description="Nom de la métrique"),
    start_date: datetime = Query(..., description="Date de début"),
    end_date: datetime = Query(..., description="Date de fin"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Export d'analytics dans différents formats
    """
    try:
        # Validation du format
        if format not in ['csv', 'json', 'excel']:
            raise HTTPException(
                status_code=400,
                detail="Format doit être: csv, json ou excel"
            )
        
        # Récupération des données
        data = await engine.query_analytics(
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date
        )
        
        # Génération du fichier selon le format
        if format == 'csv':
            output = io.StringIO()
            df = pd.DataFrame(data)
            df.to_csv(output, index=False)
            content = output.getvalue()
            
            return StreamingResponse(
                io.BytesIO(content.encode()),
                media_type='text/csv',
                headers={'Content-Disposition': f'attachment; filename=analytics_{metric_name}.csv'}
            )
        
        elif format == 'json':
            content = json.dumps(data, indent=2, default=str)
            return StreamingResponse(
                io.BytesIO(content.encode()),
                media_type='application/json',
                headers={'Content-Disposition': f'attachment; filename=analytics_{metric_name}.json'}
            )
        
        elif format == 'excel':
            output = io.BytesIO()
            df = pd.DataFrame(data)
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': f'attachment; filename=analytics_{metric_name}.xlsx'}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 📈 Endpoints de Monitoring
# ===========================

@router.get("/monitoring/health", response_model=MonitoringResponse)
async def get_system_health():
    """
    État de santé global du système ML Analytics
    """
    try:
        health_status = ml_monitor.get_monitoring_status()
        
        return MonitoringResponse(
            success=True,
            data=health_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/metrics")
async def get_metrics():
    """
    Métriques de performance en temps réel
    """
    try:
        metrics = {
            "performance": performance_monitor.get_statistics(),
            "system": ml_monitor.get_monitoring_status(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content={
            "success": True,
            "data": metrics
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/alerts")
async def get_active_alerts():
    """
    Alertes actives du système
    """
    try:
        alerts = ml_monitor.alert_manager.get_active_alerts()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "active_alerts": [alert.to_dict() for alert in alerts],
                "count": len(alerts)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str = Path(..., description="ID de l'alerte"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Résolution d'une alerte
    """
    try:
        ml_monitor.alert_manager.resolve_alert(alert_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "alert_id": alert_id,
                "status": "resolved",
                "resolved_by": current_user.get("id"),
                "resolved_at": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 🔧 Endpoints Utilitaires
# ===========================

@router.get("/stats/summary")
async def get_summary_stats(
    engine: MLAnalyticsEngine = Depends(get_ml_engine)
):
    """
    Statistiques de résumé du système ML Analytics
    """
    try:
        stats = await engine.get_summary_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": stats
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(
    cache_type: str = Query("all", description="Type de cache à vider"),
    engine: MLAnalyticsEngine = Depends(get_ml_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Vidage du cache système
    """
    try:
        cleared_items = await engine.clear_cache(cache_type)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "cache_type": cache_type,
                "cleared_items": cleared_items,
                "cleared_by": current_user.get("id"),
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Inclusion du router dans l'application principale
def include_ml_analytics_router(app):
    """Inclusion du router ML Analytics dans l'app FastAPI"""
    app.include_router(router)


# Exports publics
__all__ = [
    'router',
    'RecommendationRequest',
    'AudioAnalysisRequest',
    'ModelTrainingRequest',
    'PipelineRequest',
    'AnalyticsQuery',
    'include_ml_analytics_router'
]
