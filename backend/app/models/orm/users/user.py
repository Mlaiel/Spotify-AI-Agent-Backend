"""
User ORM Model
- Tracks all user core data, authentication, roles, audit, compliance, traceability, multi-tenancy.
- Supports advanced security, password hashing, soft-delete, GDPR/DSGVO, logging, MFA, account lockout, consent, privacy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")  # user, artist, admin, etc.
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    account_locked = Column(Boolean, default=False)
    consent_flags = Column(JSON, nullable=True)
    privacy_settings = Column(JSON, nullable=True)
    last_login = Column(DateTime, nullable=True)
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
    def create(email: str, password_hash: str, role: str = "user", **kwargs):
        return User(
            email=email,
            password_hash=password_hash,
            role=role,)
            is_active=kwargs.get("is_active", True),
            is_verified=kwargs.get("is_verified", False),
            mfa_enabled=kwargs.get("mfa_enabled", False),
            account_locked=kwargs.get("account_locked", False),
            consent_flags=kwargs.get("consent_flags"),
            privacy_settings=kwargs.get("privacy_settings"),
            last_login=kwargs.get("last_login"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "mfa_enabled": self.mfa_enabled,
            "account_locked": self.account_locked,
            "consent_flags": self.consent_flags,
            "privacy_settings": self.privacy_settings,
            "last_login": self.last_login,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
