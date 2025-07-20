"""
Collaboration Business Model
- Tracks all collaboration data (requests, matches, status, audit, compliance, traceability, multi-tenancy).
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Collaboration(Base):
    __tablename__ = "collaborations"
    id = Column(Integer, primary_key=True)
    initiator_id = Column(Integer, nullable=False, index=True)
    recipient_id = Column(Integer, nullable=False, index=True)
    status = Column(String, nullable=False, default="pending")  # pending, accepted, rejected, completed
    match_score = Column(Integer, nullable=True)
    meta_data = Column(JSON, nullable=True)  # renommé de 'metadata' à 'meta_data'
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(datetime.timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }]

    @staticmethod
    def create(initiator_id: int, recipient_id: int, **kwargs):
        return Collaboration(
            initiator_id=initiator_id,
            recipient_id=recipient_id,
            status=kwargs.get("status", "pending"),
            match_score=kwargs.get("match_score"),
            meta_data=kwargs.get("meta_data"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            tenant_id=kwargs.get("tenant_id"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "initiator_id": self.initiator_id,
            "recipient_id": self.recipient_id,
            "status": self.status,
            "match_score": self.match_score,
            "meta_data": self.meta_data,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id
        }
