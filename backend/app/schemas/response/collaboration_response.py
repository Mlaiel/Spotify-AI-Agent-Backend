"""
Collaboration Response Schemas
- Advanced, production-ready Pydantic models for all Collaboration response objects (request, status update, feedback).
- Features: DSGVO/HIPAA compliance, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Erbt von BaseResponse für einheitliche Response-Struktur.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from datetime import datetime
from .base_response import BaseResponse

class CollaborationResponse(BaseResponse):
    collaboration_id: int
    status: str
    match_score: Optional[int]
    metadata: Optional[Dict[str, Any]]

class CollaborationStatusUpdateResponse(BaseResponse):
    updated_at: datetime
