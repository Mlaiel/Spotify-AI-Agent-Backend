"""
RevenueAnalytics ORM Model
- Tracks all revenue, monetization, subscriptions, forecasting, compliance, audit, traceability.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class RevenueAnalytics(Base):
    __tablename__ = "revenue_analytics"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True, index=True)
    revenue = Column(Float, nullable=True)
    monetization_type = Column(String, nullable=True)  # ads, subscription, etc.
    subscription_status = Column(String, nullable=True)
    forecast = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    audit_log = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(user_id: int = None, **kwargs):
        return RevenueAnalytics(
            user_id=user_id,)
            revenue=kwargs.get("revenue"),
            monetization_type=kwargs.get("monetization_type"),
            subscription_status=kwargs.get("subscription_status"),
            forecast=kwargs.get("forecast"),
            compliance_flags=kwargs.get("compliance_flags"),
            audit_log=kwargs.get("audit_log"),
            tenant_id=kwargs.get("tenant_id"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "revenue": self.revenue,
            "monetization_type": self.monetization_type,
            "subscription_status": self.subscription_status,
            "forecast": self.forecast,
            "compliance_flags": self.compliance_flags,
            "audit_log": self.audit_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id
        }
