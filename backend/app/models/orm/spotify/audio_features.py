"""
AudioFeatures ORM Model
- Tracks all audio features (acousticness, danceability, energy, etc.), analytics, audit, compliance, traceability, multi-tenancy.
- Supports advanced search, analytics, soft-delete, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class AudioFeatures(Base):
    __tablename__ = "audio_features"
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, nullable=False, index=True)
    acousticness = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    key = Column(Integer, nullable=True)
    mode = Column(Integer, nullable=True)
    time_signature = Column(Integer, nullable=True)
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
    def create(track_id: int, **kwargs):
        return AudioFeatures(
            track_id=track_id,)
            acousticness=kwargs.get("acousticness"),
            danceability=kwargs.get("danceability"),
            energy=kwargs.get("energy"),
            instrumentalness=kwargs.get("instrumentalness"),
            liveness=kwargs.get("liveness"),
            loudness=kwargs.get("loudness"),
            speechiness=kwargs.get("speechiness"),
            valence=kwargs.get("valence"),
            tempo=kwargs.get("tempo"),
            key=kwargs.get("key"),
            mode=kwargs.get("mode"),
            time_signature=kwargs.get("time_signature"),
            meta_data=kwargs.get("metadata"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "track_id": self.track_id,
            "acousticness": self.acousticness,
            "danceability": self.danceability,
            "energy": self.energy,
            "instrumentalness": self.instrumentalness,
            "liveness": self.liveness,
            "loudness": self.loudness,
            "speechiness": self.speechiness,
            "valence": self.valence,
            "tempo": self.tempo,
            "key": self.key,
            "mode": self.mode,
            "time_signature": self.time_signature,
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
