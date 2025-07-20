"""
ContentAnalytics ORM Model
- Tracks all content engagement, reach, interactions, A/B tests, privacy, audit, compliance, explainability, traceability.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ContentAnalytics(Base):
    __tablename__ = "content_analytics"
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, nullable=False, index=True)
    engagement = Column(Float, nullable=True)
    reach = Column(Integer, nullable=True)
    interactions = Column(JSON, nullable=True)  # likes, shares, comments, etc.
    ab_test_group = Column(String, nullable=True)
    privacy_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(content_id: int, **kwargs):
        return ContentAnalytics(
            content_id=content_id,)
            engagement=kwargs.get("engagement"),
            reach=kwargs.get("reach"),
            interactions=kwargs.get("interactions"),
            ab_test_group=kwargs.get("ab_test_group"),
            privacy_flags=kwargs.get("privacy_flags"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "content_id": self.content_id,
            "engagement": self.engagement,
            "reach": self.reach,
            "interactions": self.interactions,
            "ab_test_group": self.ab_test_group,
            "privacy_flags": self.privacy_flags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
