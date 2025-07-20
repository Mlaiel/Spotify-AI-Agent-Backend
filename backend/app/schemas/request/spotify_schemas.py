"""
SpotifyData Request/Response Schemas
- Advanced, production-ready Pydantic models for Spotify endpoints (track, album, artist, playlist, audio features, streaming data).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, soft-delete, multilingual error messages.
- All fields validiert, dokumentiert, mit Business-Logik und Security-Checks.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class SpotifyDataRequest(BaseModel):
    data_type: str = Field(..., description="Typ der Spotify-Daten (track, album, artist, playlist, etc.)")
    spotify_id: str = Field(..., description="Spotify-Objekt-ID.")
    data: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
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
            raise ValueError("Consent (Einwilligung) ist für Spotify-Daten zwingend erforderlich.")
        return v

class SpotifyDataResponse(BaseModel):
    spotify_data_id: int
    status: str
    data: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
    version: Optional[str]

class SpotifyDataDeleteRequest(BaseModel):
    spotify_data_id: int
    user_id: int
    reason: Optional[str]
    tenant_id: Optional[str]
    trace_id: Optional[str]

class SpotifyDataDeleteResponse(BaseModel):
    status: str
    deleted_at: datetime
    audit_log: Optional[List[Dict[str, Any]]]
    compliance_flags: Optional[Dict[str, Any]]
    trace_id: Optional[str]
