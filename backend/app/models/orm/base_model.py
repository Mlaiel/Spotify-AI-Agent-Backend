"""
BaseModel for all ORM models
- Provides id, timestamps, soft-delete, audit, multi-tenancy, traceability, compliance, versioning, security, logging.
- All business models must inherit from BaseModel for consistency, auditability, and compliance.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, DateTime, Boolean, String, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    version = Column(String, default="1.0")
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.now(timezone.utc).isoformat()
        }]

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
