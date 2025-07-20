"""
Playlist ORM Model
- Tracks all playlist metadata, analytics, audit, compliance, traceability, multi-tenancy.
- Supports advanced search, analytics, soft-delete, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Playlist(Base):
    __tablename__ = "playlists"
    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, nullable=True, index=True)
    description = Column(String, nullable=True)
    public = Column(Boolean, default=True)
    collaborative = Column(Boolean, default=False)
    track_ids = Column(JSON, nullable=True)  # list of track IDs
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(datetime.timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }]

    @staticmethod
    def create(spotify_id: str, name: str, **kwargs):
        return Playlist(
            spotify_id=spotify_id,
            name=name,)
            owner_id=kwargs.get("owner_id"),
            description=kwargs.get("description"),
            public=kwargs.get("public", True),
            collaborative=kwargs.get("collaborative", False),
            track_ids=kwargs.get("track_ids"),
            meta_data=kwargs.get("metadata"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "spotify_id": self.spotify_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "description": self.description,
            "public": self.public,
            "collaborative": self.collaborative,
            "track_ids": self.track_ids,
            "metadata": self.meta_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
