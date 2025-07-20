"""
UserSpotifyData ORM Model
- Tracks all user-linked Spotify data (tokens, scopes, sync, audit, compliance, traceability, multi-tenancy).
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, token rotation, consent, privacy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UserSpotifyData(Base):
    __tablename__ = "user_spotify_data"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    spotify_user_id = Column(String, nullable=True, index=True)
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    scopes = Column(JSON, nullable=True)
    sync_status = Column(String, nullable=True)
    last_sync = Column(DateTime, nullable=True)
    consent_flags = Column(JSON, nullable=True)
    privacy_settings = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, admin_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(datetime.timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "admin_id": admin_id, "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }]

    @staticmethod
    def create(user_id: int, **kwargs):
        return UserSpotifyData(
            user_id=user_id,)
            spotify_user_id=kwargs.get("spotify_user_id"),
            access_token=kwargs.get("access_token"),
            refresh_token=kwargs.get("refresh_token"),
            token_expires_at=kwargs.get("token_expires_at"),
            scopes=kwargs.get("scopes"),
            sync_status=kwargs.get("sync_status"),
            last_sync=kwargs.get("last_sync"),
            consent_flags=kwargs.get("consent_flags"),
            privacy_settings=kwargs.get("privacy_settings"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "spotify_user_id": self.spotify_user_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expires_at": self.token_expires_at,
            "scopes": self.scopes,
            "sync_status": self.sync_status,
            "last_sync": self.last_sync,
            "consent_flags": self.consent_flags,
            "privacy_settings": self.privacy_settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
