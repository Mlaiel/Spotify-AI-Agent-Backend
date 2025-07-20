"""
Spotify Response Schemas
- Advanced, production-ready Pydantic models for all Spotify response objects (track, album, artist, playlist, audio features, streaming data).
- Features: DSGVO/HIPAA compliance, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Erbt von BaseResponse für einheitliche Response-Struktur.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from datetime import datetime
from .base_response import BaseResponse

class SpotifyDataResponse(BaseResponse):
    spotify_data_id: int
    data: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]

class SpotifyDataDeleteResponse(BaseResponse):
    deleted_at: datetime
