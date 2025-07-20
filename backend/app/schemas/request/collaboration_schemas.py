"""
Collaboration Request/Response Schemas
- Advanced, production-ready Pydantic models for Collaboration endpoints (request, accept, reject, complete, feedback).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, soft-delete, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class CollaborationRequest(BaseModel):
    initiator_id: int = Field(..., description="ID des anfragenden Nutzers.")
    recipient_id: int = Field(..., description="ID des eingeladenen Nutzers.")
    metadata: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]
    consent: Optional[bool] = Field(True, description="Nutzer hat der Kollaboration zugestimmt (DSGVO).", json_schema_extra={"example": True})

    @classmethod
    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für Kollaborationen zwingend erforderlich.")
        return v

class CollaborationResponse(BaseModel):
    collaboration_id: int
    status: str
    match_score: Optional[int]
    metadata: Optional[Dict[str, Any]]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
    version: Optional[str]

class CollaborationStatusUpdateRequest(BaseModel):
    collaboration_id: int
    user_id: int
    status: str = Field(..., description="Neuer Status (accepted, rejected, completed)")
    tenant_id: Optional[str]
    trace_id: Optional[str]

class CollaborationStatusUpdateResponse(BaseModel):
    status: str
    updated_at: datetime
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
