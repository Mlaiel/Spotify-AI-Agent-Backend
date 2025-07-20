"""
AI Tasks Package

Zentraler Einstiegspunkt für alle AI-Task-Module:
- Audioanalyse, Content-Generierung, Datenverarbeitung, Modelltraining, Recommendation-Update
- Siehe README für Details
"""
from .audio_analysis import analyze_audio_task
from .content_generation import generate_content_task
from .data_processing import process_data_task
from .model_training import train_model_task
from .recommendation_update import update_recommendations_task

__all__ = [
    "analyze_audio_task",
    "generate_content_task",
    "process_data_task",
    "train_model_task",
    "update_recommendations_task",
]
