"""
Album ORM Model
- Tracks all album metadata, release, analytics, audit, compliance, traceability, multi-tenancy.
- Supports advanced search, analytics, soft-delete, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Album(Base):
    __tablename__ = "albums"
    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    artist_id = Column(Integer, nullable=False, index=True)
    release_date = Column(DateTime, nullable=True)
    genres = Column(JSON, nullable=True)
    label = Column(String, nullable=True)
    popularity = Column(Integer, nullable=True)
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
    def create(spotify_id: str, name: str, artist_id: int, **kwargs):
        return Album(
            spotify_id=spotify_id,
            name=name,
            artist_id=artist_id,)
            release_date=kwargs.get("release_date"),
            genres=kwargs.get("genres"),
            label=kwargs.get("label"),
            popularity=kwargs.get("popularity"),
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
            "artist_id": self.artist_id,
            "release_date": self.release_date,
            "genres": self.genres,
            "label": self.label,
            "popularity": self.popularity,
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
