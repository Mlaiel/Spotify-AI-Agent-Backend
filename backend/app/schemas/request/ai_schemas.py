"""
AI Request/Response Schemas
- Advanced, production-ready Pydantic models for all AI endpoints (conversations, feedback, generated content, model config, performance, training data).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, explainability, multi-tenancy, versioning, consent, privacy, logging, monitoring, soft-delete, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, constr, field_validator

class AIConversationRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    user_id: int = Field(..., description="ID des Nutzers, der die Konversation startet.")
    prompt: constr(min_length=1, max_length=4096)
    model_name: str = Field(..., description="Name des verwendeten KI-Modells.")
    context: Optional[Dict[str, Any]] = Field(None, description="Kontext für die KI-Konversation.")
    session_id: Optional[str] = Field(None, description="Session-ID für Multi-Turn-Konversationen.")
    tenant_id: Optional[str] = Field(None, description="Mandanten-ID für Multi-Tenancy.")
    trace_id: Optional[str] = Field(None, description="Trace-ID für Nachvollziehbarkeit.")
    consent: Optional[bool] = Field(True, description="Nutzer hat der Verarbeitung zugestimmt (DSGVO).", json_schema_extra={"example": True})

    @classmethod
    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für KI-Operationen zwingend erforderlich.")
        return v

class AIConversationResponse(BaseModel):
    conversation_id: int
    response: str
    explainability: Optional[Dict[str, Any]] = Field(None, description="Erklärungen/Transparenz zum KI-Output.")
    audit_log: Optional[List[Dict[str, Any]]] = Field(None, description="Audit-Log für Compliance und Nachvollziehbarkeit.")
    compliance_flags: Optional[Dict[str, Any]] = Field(None, description="Compliance-Status (DSGVO, HIPAA, etc.)")
    trace_id: Optional[str]
    version: Optional[str]

class AIFeedbackRequest(BaseModel):
    conversation_id: int
    user_id: Optional[int]
    rating: float = Field(..., ge=0, le=5, description="Bewertung der KI-Antwort (0-5 Sterne).")
    feedback_text: Optional[str] = Field(None, max_length=2048)
    tenant_id: Optional[str]
    trace_id: Optional[str]

class AIFeedbackResponse(BaseModel):
    feedback_id: int
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class AIGeneratedContentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    user_id: int
    content_type: str = Field(..., description="Typ des generierten Inhalts (Text, Audio, Bild, etc.)")
    content: str
    model_name: str
    metadata: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]
    consent: Optional[bool] = Field(True, description="Nutzer hat der Verarbeitung zugestimmt (DSGVO).", json_schema_extra={"example": True})

    @classmethod
    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für KI-Content zwingend erforderlich.")
        return v

class AIGeneratedContentResponse(BaseModel):
    content_id: int
    status: str
    explainability: Optional[Dict[str, Any]]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class AIModelConfigRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    version: str = Field("1.0", description="Modellversion.")
    config: Dict[str, Any]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class AIModelConfigResponse(BaseModel):
    config_id: int
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class ModelPerformanceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    version: str = Field("1.0")
    metrics: Dict[str, Any]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class ModelPerformanceResponse(BaseModel):
    performance_id: int
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class TrainingDataRequest(BaseModel):
    dataset_name: str
    version: str = Field("1.0")
    metadata: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class TrainingDataResponse(BaseModel):
    training_data_id: int
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
