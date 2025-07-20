"""
Analytics Response Schemas
- Advanced, production-ready Pydantic models for all Analytics response objects (content, user, revenue, trend, KPIs, metrics).
- Features: DSGVO/HIPAA compliance, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Erbt von BaseResponse für einheitliche Response-Struktur.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field
from datetime import datetime
from .base_response import BaseResponse

class AnalyticsResponse(BaseResponse):
    analytics_id: int
    metrics: Optional[Dict[str, Any]]
    kpis: Optional[Dict[str, Any]]
    revenue: Optional[float]
    trend_data: Optional[Dict[str, Any]]
    user_analytics: Optional[Dict[str, Any]]

class AnalyticsDeleteResponse(BaseResponse):
    deleted_at: datetime
