"""
AIGeneratedContent ORM Model
- Tracks all AI-generated content (text, audio, metadata), versioning, traceability, audit, compliance, explainability.
- Supports advanced search, analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class AIGeneratedContent(Base):
    __tablename__ = "ai_generated_content"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    content_type = Column(String, nullable=False)  # text, audio, image, etc.
    content = Column(String, nullable=False)
    meta_data = Column(JSON, nullable=True)
    model_name = Column(String, nullable=False)
    version = Column(String, default="1.0")
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
    def create(user_id: int, content_type: str, content: str, model_name: str, **kwargs):
        return AIGeneratedContent(
            user_id=user_id,
            content_type=content_type,
            content=content,
            model_name=model_name,)
            meta_data=kwargs.get("metadata"),
            version=kwargs.get("version", "1.0"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content_type": self.content_type,
            "content": self.content,
            "metadata": self.meta_data,
            "model_name": self.model_name,
            "version": self.version,
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
