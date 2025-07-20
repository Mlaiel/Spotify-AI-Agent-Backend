"""
AIConversation ORM Model
- Tracks all AI chat/prompt interactions, context, user attribution, multi-tenancy, audit, compliance, explainability, traceability.
- Supports advanced search, analytics, soft-delete, versioning, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class AIConversation(Base):
    __tablename__ = "ai_conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    context = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    version = Column(String, default="1.0")
    audit_log = Column(JSON, nullable=True)
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)
    # Relationships
    feedback = relationship("AIFeedback", back_populates="conversation", cascade="all, delete-orphan")

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(user_id: int, prompt: str, response: str, model_name: str, **kwargs):
        return AIConversation(
            user_id=user_id,
            prompt=prompt,
            response=response,
            model_name=model_name,)
            context=kwargs.get("context"),
            meta_data=kwargs.get("metadata"),
            session_id=kwargs.get("session_id"),
            tenant_id=kwargs.get("tenant_id"),
            version=kwargs.get("version", "1.0"),
            audit_log=kwargs.get("audit_log"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "response": self.response,
            "model_name": self.model_name,
            "context": self.context,
            "metadata": self.meta_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "audit_log": self.audit_log,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
