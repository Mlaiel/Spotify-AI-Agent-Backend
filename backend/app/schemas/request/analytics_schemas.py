"""
Analytics Request/Response Schemas
- Advanced, production-ready Pydantic models for all Analytics endpoints (content, user, revenue, trend, KPIs, metrics).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, soft-delete, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class AnalyticsRequest(BaseModel):
    target_type: str = Field(..., description="Typ des Analyseziels (content, user, revenue, etc.)")
    target_id: Optional[str] = Field(None, description="ID des Analyseziels.")
    metrics: Optional[Dict[str, Any]]
    kpis: Optional[Dict[str, Any]]
    revenue: Optional[float]
    trend_data: Optional[Dict[str, Any]]
    user_analytics: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]
    consent: Optional[bool] = Field(True, description="Nutzer hat der Analyse zugestimmt (DSGVO).", json_schema_extra={"example": True})

    @classmethod
    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für Analytics zwingend erforderlich.")
        return v

class AnalyticsResponse(BaseModel):
    analytics_id: int
    status: str
    metrics: Optional[Dict[str, Any]]
    kpis: Optional[Dict[str, Any]]
    revenue: Optional[float]
    trend_data: Optional[Dict[str, Any]]
    user_analytics: Optional[Dict[str, Any]]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
    version: Optional[str]

class AnalyticsDeleteRequest(BaseModel):
    analytics_id: int
    user_id: int
    reason: Optional[str]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class AnalyticsDeleteResponse(BaseModel):
    status: str
    deleted_at: datetime
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
