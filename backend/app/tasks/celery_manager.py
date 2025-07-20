# 🎵 Spotify AI Agent - Celery Task Manager
# =======================================
# 
# Système complet de gestion des tâches asynchrones
# avec Celery, Redis et monitoring avancé.
#
# 🎖️ Développé par l'équipe d'experts enterprise

"""
Advanced Task Management System with Celery
==========================================

Complete asynchronous task processing system with:
- Celery workers for background processing
- Redis/RabbitMQ broker support
- Task scheduling and monitoring
- ML model training jobs
- Audio processing queues
- Real-time notifications

Authors & Roles:
- Lead Developer & AI Architect  
- Senior Backend Developer (Python/FastAPI/Django)
- DBA & Data Engineer (Redis/MongoDB)
- Microservices Architect
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from celery import Celery, Task, group, chain, chord
from celery.schedules import crontab
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue
import redis
import logging
from dataclasses import dataclass
from enum import Enum


class TaskPriority(Enum):
    """Priorités des tâches"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Status des tâches"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


@dataclass
class TaskConfig:
    """Configuration d'une tâche"""
    name: str
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 300
    queue: str = "default"
    tags: List[str] = None


class SpotifyTaskManager:
    """Gestionnaire principal des tâches Spotify AI Agent"""
    
    def __init__(self):
        self.app = self._create_celery_app()
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0))
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_celery_app(self) -> Celery:
        """Crée et configure l'application Celery"""
        app = Celery('spotify_ai_agent')
        
        # Configuration Celery
        app.conf.update(
            # Broker configuration
            broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
            result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
            
            # Task configuration
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            
            # Routing
            task_routes={
                'spotify.audio.process': {'queue': 'audio_processing'},
                'spotify.ml.train': {'queue': 'ml_training'},
                'spotify.notifications.*': {'queue': 'notifications'},
                'spotify.analytics.*': {'queue': 'analytics'},
                'spotify.user.*': {'queue': 'user_operations'},
            },
            
            # Queue configuration
            task_default_queue='default',
            task_queues=(
                Queue('default', routing_key='default'),
                Queue('audio_processing', routing_key='audio.process'),
                Queue('ml_training', routing_key='ml.train'),
                Queue('notifications', routing_key='notifications'),
                Queue('analytics', routing_key='analytics'),
                Queue('user_operations', routing_key='user'),
                Queue('high_priority', routing_key='priority.high'),
            ),
            
            # Worker configuration
            worker_concurrency=int(os.getenv('CELERY_WORKERS', 4)),
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=False,
            
            # Monitoring
            worker_send_task_events=True,
            task_send_sent_event=True,
            
            # Beat schedule (tâches périodiques)
            beat_schedule={
                'cleanup-expired-tasks': {
                    'task': 'spotify.maintenance.cleanup_expired_tasks',
                    'schedule': crontab(minute=0, hour=2),  # 2h du matin
                },
                'generate-daily-analytics': {
                    'task': 'spotify.analytics.generate_daily_report',
                    'schedule': crontab(minute=0, hour=6),  # 6h du matin
                },
                'train-recommendation-model': {
                    'task': 'spotify.ml.train_recommendation_model',
                    'schedule': crontab(minute=0, hour=3, day_of_week=1),  # Lundi 3h
                },
                'backup-user-data': {
                    'task': 'spotify.maintenance.backup_user_data',
                    'schedule': crontab(minute=0, hour=1),  # 1h du matin
                },
            },
        )
        
        return app
    
    def create_task(self, config: TaskConfig) -> Callable:
        """Créateur de tâches avec configuration"""
        def decorator(func):
            return self.app.task(
                name=config.name,
                bind=True,
                max_retries=config.max_retries,
                default_retry_delay=config.retry_delay,
                time_limit=config.timeout,
                routing_key=f"{config.queue}.{config.name}",
                tags=config.tags or []
            )(func)
        return decorator


# Instance globale du gestionnaire
task_manager = SpotifyTaskManager()


# ===== TÂCHES AUDIO PROCESSING =====

@task_manager.create_task(TaskConfig(
    name="spotify.audio.separate_track",
    priority=TaskPriority.HIGH,
    queue="audio_processing",
    timeout=600,
    tags=["audio", "spleeter"]
))
def separate_audio_track(self, track_id: str, separation_type: str = "2stems"):
    """Sépare une piste audio avec Spleeter"""
    try:
        from backend.spleeter.core.engine import SpleeterEngine
        
        # Log début
        self.logger.info(f"Début séparation audio - Track: {track_id}")
        
        # Initialiser le moteur Spleeter
        engine = SpleeterEngine()
        
        # Charger et traiter la piste
        result = engine.separate_audio(track_id, separation_type)
        
        # Sauvegarder le résultat
        self._save_separation_result(track_id, result)
        
        # Notification de fin
        notify_user_task_complete.delay(
            task_id=self.request.id,
            message="Séparation audio terminée",
            result_data=result
        )
        
        return {
            "status": "success",
            "track_id": track_id,
            "separation_type": separation_type,
            "result": result
        }
        
    except Exception as exc:
        self.logger.error(f"Erreur séparation audio: {exc}")
        self.retry(countdown=60, max_retries=3)


@task_manager.create_task(TaskConfig(
    name="spotify.audio.batch_process",
    priority=TaskPriority.NORMAL,
    queue="audio_processing",
    timeout=3600,
    tags=["audio", "batch"]
))
def batch_audio_processing(self, track_ids: List[str], processing_type: str):
    """Traitement batch de pistes audio"""
    try:
        results = []
        
        # Créer un groupe de tâches parallèles
        job = group(
            separate_audio_track.s(track_id, processing_type)
            for track_id in track_ids
        )
        
        # Exécuter en parallèle
        result = job.apply_async()
        
        # Attendre et collecter les résultats
        for res in result.get():
            results.append(res)
            
        return {
            "status": "success",
            "processed_tracks": len(track_ids),
            "results": results
        }
        
    except Exception as exc:
        self.logger.error(f"Erreur traitement batch: {exc}")
        raise


# ===== TÂCHES ML TRAINING =====

@task_manager.create_task(TaskConfig(
    name="spotify.ml.train_recommendation_model",
    priority=TaskPriority.HIGH,
    queue="ml_training",
    timeout=7200,  # 2 heures
    tags=["ml", "training", "recommendations"]
))
def train_recommendation_model(self, user_data_period: int = 30):
    """Entraîne le modèle de recommandations"""
    try:
        from backend.app.ml.recommendation_engine import RecommendationTrainer
        
        # Log début
        self.logger.info("Début entraînement modèle recommandations")
        
        # Initialiser le trainer
        trainer = RecommendationTrainer()
        
        # Charger les données utilisateur
        data = trainer.load_user_interaction_data(days=user_data_period)
        
        # Entraîner le modèle
        model_metrics = trainer.train(data)
        
        # Sauvegarder le modèle
        model_path = trainer.save_model()
        
        # Mettre à jour les métriques
        update_model_metrics.delay(
            model_type="recommendation",
            metrics=model_metrics,
            model_path=model_path
        )
        
        return {
            "status": "success",
            "model_type": "recommendation",
            "metrics": model_metrics,
            "model_path": model_path
        }
        
    except Exception as exc:
        self.logger.error(f"Erreur entraînement modèle: {exc}")
        self.retry(countdown=300, max_retries=2)


@task_manager.create_task(TaskConfig(
    name="spotify.ml.update_model_metrics",
    priority=TaskPriority.NORMAL,
    queue="ml_training",
    tags=["ml", "metrics"]
))
def update_model_metrics(self, model_type: str, metrics: Dict, model_path: str):
    """Met à jour les métriques de modèle"""
    try:
        from backend.app.models.orm.ai import ModelPerformance
        
        # Sauvegarder les métriques
        performance = ModelPerformance.create(
            model_type=model_type,
            metrics=metrics,
            model_path=model_path,
            updated_at=datetime.utcnow()
        )
        
        return {"status": "success", "performance_id": performance.id}
        
    except Exception as exc:
        self.logger.error(f"Erreur mise à jour métriques: {exc}")
        raise


# ===== TÂCHES NOTIFICATIONS =====

@task_manager.create_task(TaskConfig(
    name="spotify.notifications.notify_user_task_complete",
    priority=TaskPriority.HIGH,
    queue="notifications",
    tags=["notifications", "realtime"]
))
def notify_user_task_complete(self, task_id: str, message: str, result_data: Dict = None):
    """Notifie l'utilisateur de la fin d'une tâche"""
    try:
        from backend.app.services.notification_service import NotificationService
        
        # Envoyer notification WebSocket
        notification_service = NotificationService()
        notification_service.send_task_completion(
            task_id=task_id,
            message=message,
            data=result_data
        )
        
        return {"status": "notification_sent", "task_id": task_id}
        
    except Exception as exc:
        self.logger.error(f"Erreur envoi notification: {exc}")
        raise


# ===== UTILITAIRES DE MONITORING =====

class TaskMonitor:
    """Moniteur de tâches en temps réel"""
    
    def __init__(self):
        self.redis_client = task_manager.redis_client
        
    def get_active_tasks(self) -> List[Dict]:
        """Récupère les tâches actives"""
        active_tasks = task_manager.app.control.inspect().active()
        return active_tasks or []
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Statistiques des files d'attente"""
        stats = {}
        queues = ['default', 'audio_processing', 'ml_training', 'notifications', 'analytics']
        
        for queue in queues:
            length = self.redis_client.llen(f"celery:queue:{queue}")
            stats[queue] = length
            
        return stats


# Instance globale du moniteur
task_monitor = TaskMonitor()


# ===== API POUR INTÉGRATION FASTAPI =====

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Récupère le statut d'une tâche"""
    result = task_manager.app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
        "traceback": result.traceback
    }


def cancel_task(task_id: str) -> bool:
    """Annule une tâche"""
    task_manager.app.control.revoke(task_id, terminate=True)
    return True


# Export des fonctions principales
__all__ = [
    'task_manager',
    'task_monitor',
    'separate_audio_track',
    'batch_audio_processing',
    'train_recommendation_model',
    'notify_user_task_complete',
    'get_task_status',
    'cancel_task'
]
