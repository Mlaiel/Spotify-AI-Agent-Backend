"""
AIModelConfig ORM Model
- Tracks all AI model configurations, hyperparameters, registry, versioning, audit, security, compliance, explainability.
- Supports rollback, audit, soft-delete, GDPR/DSGVO, logging, multi-tenancy, approval workflow.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class AIModelConfig(Base):
    __tablename__ = "ai_model_config"
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    version = Column(String, default="1.0", index=True)
    hyperparameters = Column(JSON, nullable=True)
    registry_uri = Column(String, nullable=True)
    created_by = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    approval_status = Column(String, default="pending")  # pending, approved, rejected
    approval_log = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(model_name: str, version: str = "1.0", **kwargs):
        return AIModelConfig(
            model_name=model_name,
            version=version,)
            hyperparameters=kwargs.get("hyperparameters"),
            registry_uri=kwargs.get("registry_uri"),
            created_by=kwargs.get("created_by"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            approval_status=kwargs.get("approval_status", "pending"),
            approval_log=kwargs.get("approval_log"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "version": self.version,
            "hyperparameters": self.hyperparameters,
            "registry_uri": self.registry_uri,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "approval_status": self.approval_status,
            "approval_log": self.approval_log,
            "trace_id": self.trace_id
        }
