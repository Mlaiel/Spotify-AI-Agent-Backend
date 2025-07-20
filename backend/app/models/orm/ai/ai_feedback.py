"""
AIFeedback ORM Model
- Tracks user/model feedback, ratings, audit, explainability, compliance, traceability for AI interactions.
- Supports advanced analytics, soft-delete, versioning, GDPR/DSGVO, security, logging.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class AIFeedback(Base):
    __tablename__ = "ai_feedback"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("ai_conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    rating = Column(Float, nullable=True)
    feedback_text = Column(String, nullable=True)
    feedback_type = Column(String, default="user")  # user, model, audit
    explainability = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    version = Column(String, default="1.0")
    audit_log = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)
    # Relationships
    conversation = relationship("AIConversation", back_populates="feedback")

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(conversation_id: int, user_id: int = None, rating: float = None, feedback_text: str = None, **kwargs):
        return AIFeedback(
            conversation_id=conversation_id,
            user_id=user_id,
            rating=rating,
            feedback_text=feedback_text,)
            feedback_type=kwargs.get("feedback_type", "user"),
            explainability=kwargs.get("explainability"),
            compliance_flags=kwargs.get("compliance_flags"),
            tenant_id=kwargs.get("tenant_id"),
            version=kwargs.get("version", "1.0"),
            audit_log=kwargs.get("audit_log"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "feedback_type": self.feedback_type,
            "explainability": self.explainability,
            "compliance_flags": self.compliance_flags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "audit_log": self.audit_log,
            "trace_id": self.trace_id
        }
