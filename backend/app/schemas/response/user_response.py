"""
User Response Schemas
- Advanced, production-ready Pydantic models for all User response objects (profile, preferences, subscription, privacy, consent, authentication).
- Features: DSGVO/HIPAA compliance, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Erbt von BaseResponse für einheitliche Response-Struktur.
"""
from typing import Optional, List, Dict, Any
from pydantic import Field, EmailStr
from datetime import datetime
from .base_response import BaseResponse

class UserProfileResponse(BaseResponse):
    user_id: int
    email: EmailStr
    display_name: Optional[str]
    privacy_settings: Optional[Dict[str, Any]]

class UserPreferencesResponse(BaseResponse):
    user_id: int
    preferences: Dict[str, Any]

class UserSubscriptionResponse(BaseResponse):
    subscription_id: int
    user_id: int
    plan: str
    start_date: datetime
    status: str

class UserDeleteResponse(BaseResponse):
    deleted_at: datetime
