"""
Base Response Schema
- Alle Response-Objekte im Spotify AI Agent Backend erben von dieser Klasse.
- Features: DSGVO/HIPAA compliance, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Einheitliche Struktur für alle API-Responses.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    status: str = Field(..., description="Status der Response (success, error, etc.)")
    message: Optional[str] = Field(None, description="Zusätzliche Nachricht oder Fehlerbeschreibung.")
    audit_log: Optional[List[Dict[str, Any]]] = Field(None, description="Audit-Log für Compliance und Nachvollziehbarkeit.")
    compliance_flags: Optional[Dict[str, Any]] = Field(None, description="Compliance-Status (DSGVO, HIPAA, etc.)")
    trace_id: Optional[str] = Field(None, description="Trace-ID für Nachvollziehbarkeit.")
    version: Optional[str] = Field("1.0", description="API/Schema-Version.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Zeitpunkt der Response.")
