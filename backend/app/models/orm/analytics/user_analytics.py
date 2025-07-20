"""
UserAnalytics ORM Model
- Tracks all user analytics (churn, retention, segments, attribution, privacy, audit, compliance, explainability, traceability).
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UserAnalytics(Base):
    __tablename__ = "user_analytics"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    churn_score = Column(Float, nullable=True)
    retention_score = Column(Float, nullable=True)
    segments = Column(JSON, nullable=True)
    attribution = Column(JSON, nullable=True)
    privacy_flags = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    audit_log = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(user_id: int, **kwargs):
        return UserAnalytics(
            user_id=user_id,)
            churn_score=kwargs.get("churn_score"),
            retention_score=kwargs.get("retention_score"),
            segments=kwargs.get("segments"),
            attribution=kwargs.get("attribution"),
            privacy_flags=kwargs.get("privacy_flags"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            audit_log=kwargs.get("audit_log"),
            tenant_id=kwargs.get("tenant_id"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "churn_score": self.churn_score,
            "retention_score": self.retention_score,
            "segments": self.segments,
            "attribution": self.attribution,
            "privacy_flags": self.privacy_flags,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "audit_log": self.audit_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id
        }
