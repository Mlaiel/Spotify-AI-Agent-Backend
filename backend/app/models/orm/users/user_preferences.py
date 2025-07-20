"""
UserPreferences ORM Model
- Tracks all user preferences, notification settings, privacy, audit, compliance, traceability, multi-tenancy.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, versioning.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    notification_settings = Column(JSON, nullable=True)
    language = Column(String, nullable=True)
    theme = Column(String, nullable=True)
    privacy_settings = Column(JSON, nullable=True)
    preference_version = Column(String, default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, admin_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(datetime.timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "admin_id": admin_id, "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }]

    @staticmethod
    def create(user_id: int, **kwargs):
        return UserPreferences(
            user_id=user_id,)
            notification_settings=kwargs.get("notification_settings"),
            language=kwargs.get("language"),
            theme=kwargs.get("theme"),
            privacy_settings=kwargs.get("privacy_settings"),
            preference_version=kwargs.get("preference_version", "1.0"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "notification_settings": self.notification_settings,
            "language": self.language,
            "theme": self.theme,
            "privacy_settings": self.privacy_settings,
            "preference_version": self.preference_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
