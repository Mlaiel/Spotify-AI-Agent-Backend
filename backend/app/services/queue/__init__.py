"""
Spotify AI Agent – Queue Module

Created by: Achiri AI Engineering Team
Roles: Lead Dev + Architecte IA, Développeur Backend Senior, Ingénieur ML, DBA/Data Engineer, Spécialiste Sécurité, Architecte Microservices
"""
from .task_queue_service import TaskQueueService
from .job_processor import JobProcessor
from .scheduler_service import SchedulerService
from .event_publisher import EventPublisher

__version__ = "1.0.0"
__all__ = [
    "TaskQueueService",
    "JobProcessor",
    "SchedulerService",
    "EventPublisher",
]
