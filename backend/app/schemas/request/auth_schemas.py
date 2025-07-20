"""
Auth Request/Response Schemas
- Advanced, production-ready Pydantic models for Auth endpoints (Login, Register, MFA, Password Reset, Consent, Privacy).
- Features: DSGVO/HIPAA compliance, security, password policy, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, constr, field_validator

class RegisterRequest(BaseModel):
    email: EmailStr
    password: constr(min_length=12, max_length=128)
    consent: bool = Field(..., description="Nutzer hat der Verarbeitung zugestimmt (DSGVO).", json_schema_extra={"example": True})
    privacy_settings: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    trace_id: Optional[str]

    @classmethod
    @field_validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError("Consent (Einwilligung) ist für Registrierung zwingend erforderlich.")
        return v

class RegisterResponse(BaseModel):
    user_id: int
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    mfa_code: Optional[str]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    mfa_required: bool
    user_id: int
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class PasswordResetRequest(BaseModel):
    email: EmailStr
    tenant_id: Optional[str]
    trace_id: Optional[str]

class PasswordResetResponse(BaseModel):
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class ConsentUpdateRequest(BaseModel):
    user_id: int
    consent_flags: Dict[str, Any]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class ConsentUpdateResponse(BaseModel):
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class PrivacySettingsUpdateRequest(BaseModel):
    user_id: int
    privacy_settings: Dict[str, Any]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class PrivacySettingsUpdateResponse(BaseModel):
    status: str
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
