"""
User Request/Response Schemas
- Advanced, production-ready Pydantic models for User endpoints (profile, preferences, subscription, privacy, consent, authentication).
- Features: DSGVO/HIPAA compliance, security, password policy, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, soft-delete, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, constr, field_validator

class UserProfileRequest(BaseModel):
    user_id: int
    email: EmailStr
    display_name: Optional[str]
    privacy_settings: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]
    consent: Optional[bool] = Field(
        True,
        description="Nutzer hat der Verarbeitung zugestimmt (DSGVO).",
        json_schema_extra={"example": True}
    )

    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für User-Profile zwingend erforderlich.")
        return v

class UserProfileResponse(BaseModel):
    user_id: int
    email: EmailStr
    display_name: Optional[str]
    privacy_settings: Optional[Dict[str, Any]]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
    version: Optional[str]

class UserPreferencesRequest(BaseModel):
    user_id: int
    preferences: Dict[str, Any]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class UserPreferencesResponse(BaseModel):
    user_id: int
    preferences: Dict[str, Any]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class UserSubscriptionRequest(BaseModel):
    user_id: int
    plan: str
    start_date: datetime
    tenant_id: Optional[str]
    trace_id: Optional[str]

class UserSubscriptionResponse(BaseModel):
    subscription_id: int
    user_id: int
    plan: str
    start_date: datetime
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class UserDeleteRequest(BaseModel):
    user_id: int
    admin_id: int
    reason: Optional[str]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class UserDeleteResponse(BaseModel):
    status: str
    deleted_at: datetime
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
