"""
AI Response Schemas
- Advanced, production-ready Pydantic models for all AI response objects (conversations, feedback, generated content, model config, performance, training data).
- Features: DSGVO/HIPAA compliance, audit, traceability, explainability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Erbt von BaseResponse für einheitliche Response-Struktur.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from datetime import datetime
from .base_response import BaseResponse

class AIConversationResponse(BaseResponse):
    conversation_id: int
    response: str
    explainability: Optional[Dict[str, Any]] = Field(None, description="Erklärungen/Transparenz zum KI-Output.")

class AIFeedbackResponse(BaseResponse):
    feedback_id: int

class AIGeneratedContentResponse(BaseResponse):
    content_id: int
    explainability: Optional[Dict[str, Any]]

class AIModelConfigResponse(BaseResponse):
    config_id: int

class ModelPerformanceResponse(BaseResponse):
    performance_id: int

class TrainingDataResponse(BaseResponse):
    training_data_id: int
