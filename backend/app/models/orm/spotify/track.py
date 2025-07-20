"""
Track ORM Model
- Tracks all track metadata, analytics, audit, compliance, traceability, multi-tenancy.
- Supports advanced search, analytics, soft-delete, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    artist_id = Column(Integer, nullable=False, index=True)
    album_id = Column(Integer, nullable=True, index=True)
    duration_ms = Column(Integer, nullable=True)
    explicit = Column(Boolean, default=False)
    popularity = Column(Integer, nullable=True)
    audio_features = Column(JSON, nullable=True)
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
        return Track(
            spotify_id=spotify_id,
            name=name,
            artist_id=artist_id,)
            album_id=kwargs.get("album_id"),
            duration_ms=kwargs.get("duration_ms"),
            explicit=kwargs.get("explicit", False),
            popularity=kwargs.get("popularity"),
            audio_features=kwargs.get("audio_features"),
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
            "album_id": self.album_id,
            "duration_ms": self.duration_ms,
            "explicit": self.explicit,
            "popularity": self.popularity,
            "audio_features": self.audio_features,
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
