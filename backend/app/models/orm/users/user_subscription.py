"""
UserSubscription ORM Model
- Tracks all user subscriptions, plans, payments, audit, compliance, traceability, multi-tenancy.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, renewal, cancellation, consent, privacy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    plan = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    renewal_date = Column(DateTime, nullable=True)
    cancellation_date = Column(DateTime, nullable=True)
    payment_method = Column(String, nullable=True)
    payment_amount = Column(Float, nullable=True)
    consent_flags = Column(JSON, nullable=True)
    privacy_settings = Column(JSON, nullable=True)
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
    def create(user_id: int, plan: str, start_date: datetime, **kwargs):
        return UserSubscription(
            user_id=user_id,
            plan=plan,)
            status=kwargs.get("status", "active"),
            start_date=start_date,
            end_date=kwargs.get("end_date"),
            renewal_date=kwargs.get("renewal_date"),
            cancellation_date=kwargs.get("cancellation_date"),
            payment_method=kwargs.get("payment_method"),
            payment_amount=kwargs.get("payment_amount"),
            consent_flags=kwargs.get("consent_flags"),
            privacy_settings=kwargs.get("privacy_settings"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "plan": self.plan,
            "status": self.status,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "renewal_date": self.renewal_date,
            "cancellation_date": self.cancellation_date,
            "payment_method": self.payment_method,
            "payment_amount": self.payment_amount,
            "consent_flags": self.consent_flags,
            "privacy_settings": self.privacy_settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
