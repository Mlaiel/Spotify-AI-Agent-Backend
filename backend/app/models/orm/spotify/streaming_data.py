"""
StreamingData ORM Model
- Tracks all streaming events, analytics, audit, compliance, traceability, multi-tenancy.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class StreamingData(Base):
    __tablename__ = "streaming_data"
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    device = Column(String, nullable=True)
    country = Column(String, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    context = Column(JSON, nullable=True)
    event_type = Column(String, nullable=True)  # play, pause, skip, etc.
    revenue = Column(Float, nullable=True)
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
    def create(track_id: int, timestamp: datetime, **kwargs):
        return StreamingData(
            track_id=track_id,)
            user_id=kwargs.get("user_id"),
            timestamp=timestamp,
            device=kwargs.get("device"),
            country=kwargs.get("country"),
            duration_ms=kwargs.get("duration_ms"),
            context=kwargs.get("context"),
            event_type=kwargs.get("event_type"),
            revenue=kwargs.get("revenue"),
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
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "device": self.device,
            "country": self.country,
            "duration_ms": self.duration_ms,
            "context": self.context,
            "event_type": self.event_type,
            "revenue": self.revenue,
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
